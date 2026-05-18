import sqlite3
import os
import glob
import logging

_log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_NAME = os.getenv("RAG_DB_PATH") or os.path.join(DATA_DIR, "rag_app.db")
MIGRATIONS_DIR = os.path.join(BASE_DIR, "migrations")


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def run_migrations():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    migration_files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))
    applied = 0

    for filepath in migration_files:
        filename = os.path.basename(filepath)
        already_run = conn.execute(
            "SELECT id FROM schema_migrations WHERE filename = ?", (filename,)
        ).fetchone()

        if already_run:
            continue

        _log.info("applying_migration filename=%s", filename)
        try:
            with open(filepath) as f:
                sql = f.read()
            # Run each statement individually so "duplicate column" errors on
            # ALTER TABLE are skippable — happens when two migration files both
            # try to add the same column (e.g. 005_add_doc_metadata.sql and
            # 005_add_customers.sql both add doc_type to document_store).
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as se:
                    msg = str(se).lower()
                    if "duplicate column name" in msg or "already exists" in msg:
                        _log.warning("migration_skip_duplicate stmt=%s...", stmt[:60])
                    else:
                        raise
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (?)", (filename,))
            conn.commit()
            applied += 1
        except Exception as e:
            conn.close()
            raise RuntimeError(f"Migration {filename} failed: {e}")

    if applied:
        _log.info("migrations_applied count=%d", applied)
    conn.close()


def insert_application_logs(session_id, user_query, gpt_response, model,
                             confidence=0.0, escalated=False, sources="",
                             user_id="default"):
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO application_logs
            (session_id, user_id, user_query, gpt_response, model, confidence, escalated, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, user_query, gpt_response, model, confidence, int(escalated), sources)
        )
        conn.commit()


def insert_document_record(filename, user_id="default", doc_type=None, doc_date=None):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO document_store (filename, user_id, doc_type, doc_date) VALUES (?, ?, ?, ?)",
            (filename, user_id, doc_type, doc_date)
        )
        file_id = cursor.lastrowid
        conn.commit()
    return file_id


def document_exists(file_id: int, user_id: str) -> bool:
    """Check if a document exists in this workspace without fetching all records."""
    _require_user_id(user_id, "document_exists")
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id FROM document_store WHERE id=? AND user_id=? LIMIT 1",
            (file_id, user_id)
        ).fetchone()
    return row is not None


def delete_document_record(file_id, user_id="default"):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM document_store WHERE id = ? AND user_id = ?",
            (file_id, user_id)
        )
        conn.commit()
    return cursor.rowcount > 0


def _require_user_id(user_id, fn_name: str):
    """Guard against accidental None passthrough in multi-tenant code paths.

    Previously these functions defaulted to "default" when called with None,
    which silently returned another tenant's data. Callers must now pass a
    real workspace id; None or empty string raises to surface the bug early.
    """
    if not user_id:
        raise ValueError(
            f"{fn_name}: user_id must be a non-empty string "
            f"(got {user_id!r}). Pass the workspace id explicitly."
        )


def get_all_documents(user_id="default"):
    _require_user_id(user_id, "get_all_documents")
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT id, filename, user_id, upload_timestamp FROM document_store
               WHERE user_id = ? ORDER BY upload_timestamp DESC""",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_query_stats(user_id="default"):
    _require_user_id(user_id, "get_query_stats")
    with get_db_connection() as conn:

        total = conn.execute(
            "SELECT COUNT(*) as n FROM application_logs WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]

        escalated = conn.execute(
            "SELECT COUNT(*) as n FROM application_logs WHERE escalated = 1 AND user_id = ?", (user_id,)
        ).fetchone()["n"]

        avg_conf = conn.execute(
            "SELECT AVG(confidence) as avg FROM application_logs WHERE user_id = ?", (user_id,)
        ).fetchone()["avg"] or 0.0

        recent = [row["user_query"] for row in conn.execute(
            "SELECT user_query FROM application_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
            (user_id,)
        ).fetchall()]

        gaps = [row["user_query"] for row in conn.execute(
            """SELECT user_query FROM application_logs
            WHERE escalated = 1 AND user_id = ?
            ORDER BY created_at DESC LIMIT 10""",
            (user_id,)
        ).fetchall()]
    return {
        "total_queries": total,
        "escalated_count": escalated,
        "avg_confidence": round(avg_conf, 2),
        "top_questions": recent,
        "unanswered_topics": gaps
    }


def get_audit_log(user_id="default", limit=100):
    _require_user_id(user_id, "get_audit_log")
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT session_id, user_query, gpt_response, model,
                    confidence, escalated, sources, created_at
            FROM application_logs
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
    return [dict(row) for row in rows]


def insert_brief_log(customer_id: str, query: str, brief_json: str,
                     faithfulness_score: float = 0.0, loop_count: int = 0):
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO brief_logs (customer_id, query, brief_json, faithfulness_score, loop_count)
               VALUES (?, ?, ?, ?, ?)""",
            (customer_id, query, brief_json, faithfulness_score, loop_count),
        )
        conn.commit()


# ── Customer management ───────────────────────────────────────────────────────

def create_customer(name: str, slug: str, fde_user_id: str) -> dict:
    """Create a new customer workspace. Raises sqlite3.IntegrityError if slug exists."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO customers (name, slug, fde_user_id) VALUES (?, ?, ?)",
            (name, slug.lower().strip(), fde_user_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, name, slug, fde_user_id, last_call_date, created_at FROM customers WHERE id=?",
            (cursor.lastrowid,)
        ).fetchone()
    return dict(row)


def get_customers(fde_user_id: str) -> list:
    _require_user_id(fde_user_id, "get_customers")
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, slug, last_call_date, created_at FROM customers WHERE fde_user_id=? ORDER BY name",
            (fde_user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_customer_by_slug(slug: str, fde_user_id: str) -> "dict | None":
    _require_user_id(fde_user_id, "get_customer_by_slug")
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, name, slug, fde_user_id, last_call_date, created_at FROM customers WHERE slug=? AND fde_user_id=?",
            (slug.lower().strip(), fde_user_id)
        ).fetchone()
    return dict(row) if row else None


def delete_customer(slug: str, fde_user_id: str):
    """Delete a customer and all associated DB records.

    Returns list of document file_ids that were deleted (caller should clean Chroma),
    or None if the customer was not found / not owned by fde_user_id.
    """
    _require_user_id(fde_user_id, "delete_customer")
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id FROM customers WHERE slug=? AND fde_user_id=?",
            (slug, fde_user_id)
        ).fetchone()
        if row is None:
            return None
        numeric_id = row["id"]

        doc_rows = conn.execute(
            "SELECT id FROM document_store WHERE user_id=?", (slug,)
        ).fetchall()
        file_ids = [r["id"] for r in doc_rows]

        conn.execute("DELETE FROM people WHERE customer_id=?", (numeric_id,))
        conn.execute("DELETE FROM document_store WHERE user_id=?", (slug,))
        conn.execute("DELETE FROM customers WHERE id=?", (numeric_id,))
        conn.commit()
    return file_ids


def update_last_call_date(customer_id: int, date_str: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE customers SET last_call_date=? WHERE id=?",
            (date_str, customer_id)
        )
        conn.commit()


def get_corpus_health(customer_id: str) -> dict:
    """Return per-doc-type health for a customer workspace."""
    _require_user_id(customer_id, "get_corpus_health")
    from datetime import datetime
    from utils.doc_type_utils import VALID_DOC_TYPES
    DOC_TYPES = sorted(VALID_DOC_TYPES)
    STALE_DAYS = 30
    today = datetime.now().strftime("%Y-%m-%d")

    # Single connection for both queries to use consistent WAL snapshot
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT doc_type, MAX(doc_date) as last_upload, COUNT(*) as count
               FROM document_store
               WHERE user_id=? AND doc_type IS NOT NULL
               GROUP BY doc_type""",
            (customer_id,)
        ).fetchall()

        last_call_row = conn.execute(
            "SELECT last_call_date FROM customers WHERE slug=?",
            (customer_id,)
        ).fetchone()

    by_type = {r["doc_type"]: dict(r) for r in rows}
    last_call = dict(last_call_row)["last_call_date"] if last_call_row else None

    health = {}
    missing = []
    overall = "current"

    for dt in DOC_TYPES:
        if dt not in by_type:
            # Missing means not yet uploaded — only mark stale for docs that exist but are old
            health[dt] = {"last_upload": None, "count": 0, "status": "missing"}
            missing.append(dt)
        else:
            info = by_type[dt]
            last = info["last_upload"] or today
            try:
                days_old = (datetime.strptime(today, "%Y-%m-%d") -
                            datetime.strptime(last, "%Y-%m-%d")).days
            except Exception:
                days_old = 0
            status = "current" if days_old <= STALE_DAYS else "stale"
            if status == "stale":
                overall = "stale"
            health[dt] = {"last_upload": last, "count": info["count"], "status": status}

    return {
        "doc_types": health,
        "overall": overall if any(v["status"] != "missing" for v in health.values()) else "empty",
        "last_call_date": last_call,
        "missing_doc_types": missing,
    }


# ── Stakeholder / people ──────────────────────────────────────────────────────

def add_person(customer_id: int, name: str, role: str = None, email: str = None) -> dict:
    with get_db_connection() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO people (customer_id, name, role, email) VALUES (?, ?, ?, ?)",
                (customer_id, name, role, email)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise  # caller handles as 409
        row = conn.execute(
            "SELECT id, customer_id, name, role, email FROM people WHERE id=?",
            (cursor.lastrowid,)
        ).fetchone()
    return dict(row)


def get_people(customer_id: int) -> list:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, role, email, created_at FROM people WHERE customer_id=? ORDER BY name",
            (customer_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_person_by_id(person_id: int, customer_id: str) -> "dict | None":
    """Return person record only when they belong to the specified customer."""
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT p.id, p.name, p.role, p.email
               FROM people p
               JOIN customers c ON c.id = p.customer_id
               WHERE p.id=? AND c.slug=?""",
            (person_id, customer_id)
        ).fetchone()
    return dict(row) if row else None


# ── Document versioning ───────────────────────────────────────────────────────

def set_latest_version_flag(customer_id: str, doc_type: str, new_file_id: int) -> None:
    """Flip is_latest_version=0 on all prior uploads of this doc_type for this customer,
    then set is_latest_version=1 on the new file."""
    version_group = f"{customer_id}::{doc_type}"
    with get_db_connection() as conn:
        conn.execute(
            """UPDATE document_store SET is_latest_version=0
               WHERE user_id=? AND doc_type=? AND id!=?""",
            (customer_id, doc_type, new_file_id)
        )
        conn.execute(
            """UPDATE document_store SET is_latest_version=1, doc_version_group=?
               WHERE id=?""",
            (version_group, new_file_id)
        )
        conn.commit()


# ── Brief history and feedback ────────────────────────────────────────────────

def get_brief_history(customer_id: str, limit: int = 20) -> list:
    _require_user_id(customer_id, "get_brief_history")
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT id, customer_id, query, faithfulness_score, loop_count, created_at
               FROM brief_logs WHERE customer_id=? ORDER BY created_at DESC LIMIT ?""",
            (customer_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_brief_feedback(brief_log_id: int, customer_id: str,
                           section: str, rating: int,
                           flagged_claim: str = None) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO brief_feedback (brief_log_id, customer_id, section, rating, flagged_claim)
               VALUES (?, ?, ?, ?, ?)""",
            (brief_log_id, customer_id, section, rating, flagged_claim)
        )
        conn.commit()


