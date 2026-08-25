from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from main import app

client = TestClient(app)

def test_generate_dataset():
    response = client.post("/generate-dataset", json={"source_text": "AI models need testing.", "num_samples": 2})
    assert response.status_code == 200
    assert "dataset" in response.json()
    dataset = response.json()["dataset"]
    assert len(dataset) == 2
    assert "AI models need testing." in dataset[0]["context"]
