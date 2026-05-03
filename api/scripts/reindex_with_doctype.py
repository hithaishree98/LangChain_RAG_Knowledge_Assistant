"""
scripts/reindex_with_doctype.py — One-time migration helper.

Backfills `doc_type` metadata on chunks that were indexed before the Phase 1
upload-field change. Reads every document in the DB, infers doc_type from
filename keywords, and re-stamps the chunk metadata in both vectorstore and
parent_store.

Usage:
    cd api
    python scripts/reindex_with_doctype.py [--dry-run] [--user-id USER_ID]

Options:
    --dry-run       Print what would be changed without writing anything.
    --user-id       Only process documents for this workspace. Default: all.

How it works:
  1. Reads all document records from SQLite via db_utils.get_all_documents().
  2. For each document filename, infers doc_type via _infer_doc_type().
  3. Fetches all chunks for that file_id from ChromaDB.
  4. Updates any chunk whose doc_type is missing or "unknown" with the inferred value.

Existing doc_type values that are already correct are NOT overwritten — this
script only fills in gaps. Re-run safely at any time.

Note: if you want to re-ingest a document with full Phase 2 metadata (reporter,
assignee, meeting_type, etc.), delete and re-upload it via the API — this script
only patches doc_type, not the deeper structured fields.
"""

import argparse
import os
import sys

# Allow running from the scripts/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from db_utils import get_all_documents
from chroma_utils import vectorstore, parent_store
from utils.doc_type_utils import infer_doc_type as _infer_doc_type_shared


def _infer_doc_type(filename: str) -> str:
    """Infer doc_type for backfill — falls back to 'unknown' when ambiguous."""
    result = _infer_doc_type_shared(filename)
    if result is None:
        # .json without keywords: default to "ticket" for backfill purposes
        import os as _os
        ext = _os.path.splitext(filename.lower())[1]
        return "ticket" if ext == ".json" else "unknown"
    return result


def _get_chunks_for_file(collection, file_id: int, user_id: str | None):
    """Return (ids, metadatas) for all chunks belonging to file_id."""
    where: dict = {"file_id": {"$eq": file_id}}
    if user_id:
        where = {"$and": [where, {"user_id": {"$eq": user_id}}]}
    result = collection.get(where=where, include=["metadatas"])
    return result.get("ids") or [], result.get("metadatas") or []


def run(dry_run: bool = False, filter_user_id: str | None = None):
    print(f"{'[DRY RUN] ' if dry_run else ''}Starting doc_type backfill...")

    docs = get_all_documents(user_id=filter_user_id)
    if not docs:
        print("No documents found. Nothing to do.")
        return

    total_updated = 0
    total_skipped = 0

    for doc in docs:
        file_id = doc["id"]
        filename = doc["filename"]
        user_id = doc.get("user_id", "default")

        inferred = _infer_doc_type(filename)

        for store_label, collection in [
            ("vectorstore", vectorstore._collection),
            ("parent_store", parent_store._collection),
        ]:
            ids, metas = _get_chunks_for_file(collection, file_id, user_id)
            if not ids:
                continue

            ids_to_update = []
            new_metas = []
            for cid, meta in zip(ids, metas):
                existing = (meta or {}).get("doc_type", "")
                if existing and existing != "unknown":
                    total_skipped += 1
                    continue
                updated_meta = dict(meta or {})
                updated_meta["doc_type"] = inferred
                ids_to_update.append(cid)
                new_metas.append(updated_meta)

            if ids_to_update:
                print(f"  {store_label}: file_id={file_id} ({filename}) "
                      f"→ doc_type={inferred} [{len(ids_to_update)} chunks]")
                if not dry_run:
                    collection.update(ids=ids_to_update, metadatas=new_metas)
                total_updated += len(ids_to_update)

    print(f"\nDone. Updated: {total_updated} chunks, Skipped (already tagged): {total_skipped}.")
    if dry_run:
        print("(Dry run — no changes were written.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill doc_type on existing chunks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing to ChromaDB.")
    parser.add_argument("--user-id", default=None,
                        help="Only process documents for this workspace.")
    args = parser.parse_args()
    run(dry_run=args.dry_run, filter_user_id=args.user_id)
