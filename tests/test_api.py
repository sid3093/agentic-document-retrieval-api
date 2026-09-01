from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_schema_validation():
    # Sending invalid body should trigger 422 Unprocessable Entity
    response = client.post("/ask", json={})
    assert response.status_code == 422