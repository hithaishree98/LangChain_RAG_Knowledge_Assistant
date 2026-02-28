import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data

def test_list_docs():
    response = client.get("/list-docs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_analytics():
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data

def test_audit_log():
    response = client.get("/audit-log")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_delete_nonexistent_doc():
    response = client.post("/delete-doc", json={"file_id": 99999})
    assert response.status_code == 200
    assert "error" in response.json()