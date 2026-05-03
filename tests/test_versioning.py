"""Document versioning tests.

Verifies that set_latest_version_flag() correctly promotes a new document and
demotes previous versions, with cross-doc-type independence.
"""
import pytest
import os
import sys
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


class TestDocumentVersioning:
    def test_second_upload_flips_old_to_not_latest(self, tmp_db):
        import sqlite3
        import api.db_utils as db_utils
        importlib.reload(db_utils)

        id1 = db_utils.insert_document_record(
            "2024-09-01_commitments_tracker.csv",
            user_id="cascadia-1", doc_type="commitment_tracker", doc_date="2024-09-01",
        )
        db_utils.set_latest_version_flag("cascadia-1", "commitment_tracker", id1)

        id2 = db_utils.insert_document_record(
            "2024-10-15_commitments_tracker.csv",
            user_id="cascadia-1", doc_type="commitment_tracker", doc_date="2024-10-15",
        )
        db_utils.set_latest_version_flag("cascadia-1", "commitment_tracker", id2)

        conn = sqlite3.connect(tmp_db)
        r1 = conn.execute(
            "SELECT is_latest_version FROM document_store WHERE id=?", (id1,)
        ).fetchone()
        r2 = conn.execute(
            "SELECT is_latest_version FROM document_store WHERE id=?", (id2,)
        ).fetchone()
        conn.close()

        assert r1[0] == 0  # old version flipped to not-latest
        assert r2[0] == 1  # new version is latest

    def test_different_doc_types_independent(self, tmp_db):
        import sqlite3
        import api.db_utils as db_utils
        importlib.reload(db_utils)

        tid = db_utils.insert_document_record(
            "2024-10-01_tickets_open.csv",
            user_id="cascadia-1", doc_type="ticket", doc_date="2024-10-01",
        )
        cid = db_utils.insert_document_record(
            "2024-10-15_commitments_tracker.csv",
            user_id="cascadia-1", doc_type="commitment_tracker", doc_date="2024-10-15",
        )
        db_utils.set_latest_version_flag("cascadia-1", "ticket", tid)
        db_utils.set_latest_version_flag("cascadia-1", "commitment_tracker", cid)

        # Upload new tickets — should NOT affect the commitments record
        tid2 = db_utils.insert_document_record(
            "2024-10-20_tickets_open.csv",
            user_id="cascadia-1", doc_type="ticket", doc_date="2024-10-20",
        )
        db_utils.set_latest_version_flag("cascadia-1", "ticket", tid2)

        conn = sqlite3.connect(tmp_db)
        cr = conn.execute(
            "SELECT is_latest_version FROM document_store WHERE id=?", (cid,)
        ).fetchone()
        conn.close()

        assert cr[0] == 1  # commitments version unaffected by tickets update


class TestLatestVersionRetrieval:
    """Verify that retrieval queries filter on is_latest_version=1."""

    def test_get_latest_chunks_filter_includes_is_latest_version(self):
        """get_latest_chunks_by_doctype must pass is_latest_version=1 to Chroma."""
        from unittest.mock import MagicMock, patch

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": [], "metadatas": []}

        with patch("chroma_utils.vectorstore") as mock_vs:
            mock_vs._collection = mock_collection
            from chroma_utils import get_latest_chunks_by_doctype
            get_latest_chunks_by_doctype("cascadia-inc", "commitment_tracker")

        assert mock_collection.get.called
        where = mock_collection.get.call_args.kwargs.get("where") or {}
        where_str = str(where)
        assert "is_latest_version" in where_str
        assert "1" in where_str

    def test_hybrid_retriever_filter_includes_is_latest_version(self):
        """HybridRetriever must pass is_latest_version=1 in the dense filter."""
        from unittest.mock import MagicMock, patch

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch("chroma_utils.vectorstore") as mock_vs:
            mock_vs.as_retriever.return_value = mock_retriever
            mock_vs._collection.get.return_value = {"documents": [], "metadatas": []}
            from chroma_utils import HybridRetriever
            r = HybridRetriever(user_id="cascadia-inc", k_dense=4, k_bm25=4, k_rerank=2)
            r._get_relevant_documents("what are the open commitments?")

        assert mock_vs.as_retriever.called
        search_kwargs = mock_vs.as_retriever.call_args.kwargs.get("search_kwargs", {})
        filt = search_kwargs.get("filter", {})
        filt_str = str(filt)
        assert "is_latest_version" in filt_str

    def test_get_chunks_since_date_filter_includes_is_latest_version(self):
        """get_chunks_since_date must filter on is_latest_version=1."""
        from unittest.mock import MagicMock, patch

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": [], "metadatas": []}

        with patch("chroma_utils.vectorstore") as mock_vs:
            mock_vs._collection = mock_collection
            from chroma_utils import get_chunks_since_date
            get_chunks_since_date("cascadia-inc", "2024-09-01")

        where = mock_collection.get.call_args.kwargs.get("where") or {}
        assert "is_latest_version" in str(where)
