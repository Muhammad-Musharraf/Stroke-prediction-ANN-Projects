# tests/test_api.py
from fastapi.testclient import TestClient
from FastApi.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200