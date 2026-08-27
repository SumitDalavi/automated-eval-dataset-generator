import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys

# Mock openai BEFORE any imports
mock_openai_module = MagicMock()
mock_openai_client = MagicMock()
mock_openai_module.OpenAI.return_value = mock_openai_client
sys.modules["openai"] = mock_openai_module

# Ensure OPENAI_API_KEY is present
os.environ["OPENAI_API_KEY"] = "fake-key"

from src.main import app
from src.generator import DatasetSpec, generate_dataset
from src.strategies.qa_strategy import QAStrategy
from src.strategies.adversarial_strategy import AdversarialStrategy
from src.strategies.base import EvalSample
from src.exporters.jsonl_exporter import export_jsonl
from src.validators.quality_check import validate_and_deduplicate

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_generate_endpoint_success(tmp_path):
    # Mock strategies to avoid openai calls in integration test
    with patch("src.generator._STRATEGIES", {
        "qa": QAStrategy(),
        "adversarial": AdversarialStrategy(),
    }):
        import src.strategies.qa_strategy as qs
        import src.strategies.adversarial_strategy as ads
        qs._LLM = False
        ads._LLM = False
        
        # Test bad difficulty split
        res = client.post("/api/v1/generate", json={
            "topic": "test",
            "total_samples": 10,
            "strategies": ["qa"],
            "difficulty_split": {"easy": 0.5, "hard": 0.4} # Sums to 0.9 != 1.0
        })
        assert res.status_code == 400
        
        # Test success
        res = client.post("/api/v1/generate", json={
            "topic": "test",
            "total_samples": 10,
            "strategies": ["qa", "adversarial"],
            "difficulty_split": {"easy": 0.5, "hard": 0.5},
            "export_format": "openai"
        })
        assert res.status_code == 200
        data = res.json()
        assert "dataset_id" in data
        assert data["total_generated"] > 0

def test_qa_strategy():
    import src.strategies.qa_strategy as qs
    qs._LLM = True
    qa = QAStrategy()
    
    # Mock LLM Success
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='[{"question": "q1", "answer": "a1", "category": "c1"}]'))]
    mock_openai_client.chat.completions.create.return_value = mock_resp
    
    samples = qa.generate("topic", 1, "easy")
    assert len(samples) == 1
    assert samples[0].input == "q1"
    
    # Mock LLM Bad JSON
    mock_resp.choices = [MagicMock(message=MagicMock(content='bad json'))]
    samples = qa.generate("topic", 1, "easy")
    assert len(samples) == 1  # Fallback to template
    assert "source" in samples[0].metadata
    assert samples[0].metadata["source"] == "template"

    # Mock LLM Exception is NOT caught in qa_strategy natively? 
    # Actually wait, in QAStrategy, exception is caught?
    # No, but let's test _LLM = False
    qs._LLM = False
    samples = qa.generate("topic", 1, "medium")
    assert len(samples) == 1
    assert samples[0].difficulty == "medium"
    
    # Test all difficulties
    samples = qa.generate("topic", 1, "hard")
    samples = qa.generate("topic", 1, "adversarial")
    assert len(samples) == 1

def test_adversarial_strategy():
    import src.strategies.adversarial_strategy as ads
    ads._LLM = True
    ad = AdversarialStrategy()
    
    # Mock LLM Success
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='[{"input": "in", "expected_output": "out", "rationale": "r"}]'))]
    mock_openai_client.chat.completions.create.return_value = mock_resp
    
    samples = ad.generate("topic", 2)
    assert len(samples) == 2 # 1 per category, and we mocked loop
    
    # Mock LLM Exception (catches and uses template)
    mock_openai_client.chat.completions.create.side_effect = Exception("err")
    samples = ad.generate("topic", 2)
    assert len(samples) == 2
    assert samples[0].metadata["source"] == "template"
    mock_openai_client.chat.completions.create.side_effect = None

def test_quality_check():
    s1 = EvalSample(id="1", input="This is a long input that is valid.", expected_output="Valid output", difficulty="easy", category="c", metadata={})
    s2 = EvalSample(id="2", input="Short", expected_output="Short", difficulty="easy", category="c", metadata={})
    s3 = EvalSample(id="3", input="This is a long input that is valid.", expected_output="Duplicate", difficulty="easy", category="c", metadata={}) # near dup of s1
    
    cleaned, report = validate_and_deduplicate([s1, s2, s3], min_input_len=10, min_output_len=5)
    assert len(cleaned) == 1
    assert report["removed_too_short"] == 1
    assert report["removed_duplicates"] == 1

    # Empty sets jaccard
    from src.validators.quality_check import _jaccard
    assert _jaccard(set(), set()) == 0.0

def test_jsonl_exporter(tmp_path):
    samples = [
        EvalSample(id="1", input="in1", expected_output="out1", difficulty="easy", category="c1", metadata={"m": "v"}),
        EvalSample(id="2", input="in2", expected_output="out2", difficulty="hard", category="c2", metadata={"m": "v"}),
    ]
    p1 = str(tmp_path / "1.jsonl")
    export_jsonl(samples, p1, format="openai")
    assert os.path.exists(p1)
    
    p2 = str(tmp_path / "2.jsonl")
    export_jsonl(samples, p2, format="alpaca")
    assert os.path.exists(p2)
    
    p3 = str(tmp_path / "3.jsonl")
    export_jsonl(samples, p3, format="raw")
    assert os.path.exists(p3)

def test_generator_bad_strategy():
    spec = DatasetSpec(topic="test", total_samples=5, strategies=["bad_strat", "qa"])
    import src.strategies.qa_strategy as qs
    qs._LLM = False
    res = generate_dataset(spec)
    assert res.total_generated > 0

def test_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.strategies.qa_strategy as qs
    qs._LLM = False
    from src.main import health
    res = health()
    assert res["llm_available"] is False
