"""Tests for document ingestion parsers and filename utilities."""
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


class TestFilenameValidation:
    def test_valid_iso_date(self):
        from utils.doc_type_utils import validate_filename
        valid, err = validate_filename("2024-09-15_transcript_status-call.txt")
        assert valid, err

    def test_valid_quarter_format(self):
        from utils.doc_type_utils import validate_filename
        valid, err = validate_filename("2024-Q3_qbr_deck.pdf")
        assert valid, err

    def test_invalid_no_date(self):
        from utils.doc_type_utils import validate_filename
        valid, err = validate_filename("transcript_status-call.txt")
        assert not valid
        assert "YYYY-MM-DD" in err

    def test_invalid_wrong_format(self):
        from utils.doc_type_utils import validate_filename
        valid, err = validate_filename("my_notes.txt")
        assert not valid

    def test_extract_date_iso(self):
        from utils.doc_type_utils import extract_date_from_filename
        assert extract_date_from_filename("2024-09-15_transcript_call.txt") == "2024-09-15"

    def test_extract_date_quarter(self):
        from utils.doc_type_utils import extract_date_from_filename
        result = extract_date_from_filename("2024-Q3_qbr_deck.pdf")
        assert result == "2024-07-01"

    def test_extract_date_none(self):
        from utils.doc_type_utils import extract_date_from_filename
        assert extract_date_from_filename("notes.txt") is None


class TestTicketCSVParser:
    def test_parse_standard_csv(self, sample_tickets_csv_file):
        from ingestion.ticket_csv_parser import parse_csv
        tickets = parse_csv(sample_tickets_csv_file)
        assert len(tickets) == 3
        assert tickets[0].ticket_id == "TICK-4521"
        assert tickets[0].status == "open"
        assert tickets[0].priority == "P0"

    def test_parse_missing_optional_columns(self, tmp_path):
        from ingestion.ticket_csv_parser import parse_csv
        csv_content = "ticket_id,summary,status\nTICK-001,Test issue,open\n"
        f = tmp_path / "2024-10-01_tickets_open.csv"
        f.write_text(csv_content)
        tickets = parse_csv(str(f))
        assert len(tickets) == 1
        assert tickets[0].ticket_id == "TICK-001"
        assert tickets[0].priority == "normal"  # default

    def test_parse_missing_required_column_raises(self, tmp_path):
        from ingestion.ticket_csv_parser import parse_csv
        csv_content = "summary,status\nTest issue,open\n"
        f = tmp_path / "2024-10-01_tickets_open.csv"
        f.write_text(csv_content)
        with pytest.raises(ValueError, match="(?i)missing"):
            parse_csv(str(f))

    def test_date_normalization(self, tmp_path):
        from ingestion.ticket_csv_parser import parse_csv
        csv_content = "ticket_id,summary,status,created_date\nTICK-001,Test,open,10/15/2024\n"
        f = tmp_path / "2024-10-01_tickets_open.csv"
        f.write_text(csv_content)
        tickets = parse_csv(str(f))
        assert tickets[0].created_date == "2024-10-15"


class TestCommitmentCSVParser:
    def test_parse_standard_csv(self, sample_commitments_csv_file):
        from ingestion.commitment_parser import parse_csv
        commitments = parse_csv(sample_commitments_csv_file)
        assert len(commitments) == 3
        assert commitments[0].description == "SSO integration delivery"
        assert commitments[0].status == "open"

    def test_is_slipped_computed(self, tmp_path):
        from ingestion.commitment_parser import parse_csv
        csv_content = (
            "commitment,promised_date,current_target_date,status,owner,customer_aware\n"
            "SSO,2024-10-01,2024-11-01,open,dev,true\n"
        )
        f = tmp_path / "2024-10-15_commitments_tracker.csv"
        f.write_text(csv_content)
        commitments = parse_csv(str(f))
        assert commitments[0].is_slipped is True  # target > promised


class TestVersioning:
    def test_upload_sets_latest_true(self, tmp_db, tmp_path):
        import sqlite3
        import importlib
        import api.db_utils as db_utils
        importlib.reload(db_utils)
        file_id = db_utils.insert_document_record(
            "2024-10-15_tickets_open.csv",
            user_id="cascadia-1",
            doc_type="tickets",
            doc_date="2024-10-15",
        )
        db_utils.set_latest_version_flag("cascadia-1", "tickets", file_id)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT is_latest_version FROM document_store WHERE id=?", (file_id,)
        ).fetchone()
        assert row[0] == 1
        conn.close()
