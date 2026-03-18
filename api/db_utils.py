import sqlite3
import os
import hashlib
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_NAME = os.path.join(DATA_DIR, "rag_app.db")
MIGRATIONS_DIR = os.path.join(BASE_DIR, "migrations")


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
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

        print(f"[db] Applying {filename}")
        try:
            with open(filepath) as f:
                conn.executescript(f.read())
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (?)", (filename,))
            conn.commit()
            applied += 1
        except Exception as e:
            conn.close()
            raise RuntimeError(f"Migration {filename} failed: {e}")

    if applied:
        print(f"[db] {applied} migration(s) applied")
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


def get_chat_history(session_id, user_id="default"):
    with get_db_connection() as conn:
        rows = conn.execute(
        """SELECT user_query, gpt_response FROM application_logs
           WHERE session_id = ? AND user_id = ?
           ORDER BY created_at""",
        (session_id, user_id)
        ).fetchall()
    messages = []
    for row in rows:
        messages.extend([
            {"role": "human", "content": row["user_query"]},
            {"role": "ai", "content": row["gpt_response"]}
        ])
    return messages


def insert_document_record(filename, user_id="default"):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO document_store (filename, user_id) VALUES (?, ?)",
            (filename, user_id)
        )
        file_id = cursor.lastrowid
        conn.commit()
    return file_id


def delete_document_record(file_id, user_id="default"):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM document_store WHERE id = ? AND user_id = ?",
            (file_id, user_id)
        )
        conn.commit()
    return cursor.rowcount > 0


def get_all_documents(user_id="default"):
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT id, filename, user_id, upload_timestamp FROM document_store
               WHERE user_id = ? ORDER BY upload_timestamp DESC""",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_query_stats(user_id="default"):
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


def generate_user_id(workspace: str, passkey: str) -> str:
    raw = f"{workspace.strip().lower()}:{passkey.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


