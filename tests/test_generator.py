"""Tests for the automated eval dataset generator."""
import pytest
import os
import json
from unittest.mock import patch

from src.strategies.base import EvalSample
from src.strategies.qa_strategy import QAStrategy
from src.strategies.adversarial_strategy import AdversarialStrategy
from src.exporters.jsonl_exporter import export_jsonl
from src.validators.quality_check import validate_and_deduplicate, _jaccard
from src.generator import DatasetSpec, generate_dataset


# ── Base Tests ────────────────────────────────────────────────────────────────

def test_eval_sample_creation():
    s = EvalSample(id="1", input="Q?", expected_output="A.", difficulty="easy", category="test")
    assert s.id == "1"
    assert s.difficulty == "easy"
    assert s.metadata == {}


# ── QA Strategy Tests ─────────────────────────────────────────────────────────

def test_qa_strategy_template_easy():
    strategy = QAStrategy()
    with patch("src.strategies.qa_strategy._LLM", False):
        samples = strategy.generate("Redis", n_samples=3, difficulty="easy")
    assert len(samples) == 3
    for s in samples:
        assert s.input != ""
        assert s.expected_output != ""
        assert s.difficulty == "easy"


def test_qa_strategy_all_difficulties():
    strategy = QAStrategy()
    with patch("src.strategies.qa_strategy._LLM", False):
        for diff in ["easy", "medium", "hard", "adversarial"]:
            samples = strategy.generate("Kubernetes", n_samples=2, difficulty=diff)
            assert len(samples) == 2
            assert all(s.difficulty == diff for s in samples)


# ── Adversarial Strategy Tests ────────────────────────────────────────────────

def test_adversarial_strategy_template():
    strategy = AdversarialStrategy()
    with patch("src.strategies.adversarial_strategy._LLM", False):
        samples = strategy.generate("LLMs", n_samples=3)
    assert len(samples) == 3
    for s in samples:
        assert s.difficulty == "adversarial"


# ── Validator Tests ───────────────────────────────────────────────────────────

def test_jaccard_similarity():
    assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert 0 < _jaccard({"a", "b", "c"}, {"b", "c", "d"}) < 1.0


def test_deduplication_removes_copies():
    samples = [
        EvalSample(id="1", input="What is Redis?", expected_output="A cache.", difficulty="easy", category="redis"),
        EvalSample(id="2", input="What is Redis?", expected_output="An in-memory store.", difficulty="easy", category="redis"),
        EvalSample(id="3", input="How does Kafka work?", expected_output="Message queue.", difficulty="medium", category="kafka"),
    ]
    cleaned, report = validate_and_deduplicate(samples, similarity_threshold=0.85)
    assert report["removed_duplicates"] >= 1
    assert len(cleaned) < len(samples)


def test_length_filter_removes_short():
    samples = [
        EvalSample(id="1", input="Hi", expected_output="A.", difficulty="easy", category="x"),
        EvalSample(id="2", input="What is the meaning of distributed computing?", expected_output="Distributed computing is...", difficulty="easy", category="x"),
    ]
    cleaned, report = validate_and_deduplicate(samples, min_input_len=10)
    assert report["removed_too_short"] == 1
    assert len(cleaned) == 1


# ── Exporter Tests ────────────────────────────────────────────────────────────

def test_export_jsonl_openai_format(tmp_path):
    samples = [
        EvalSample(id="1", input="What is RAG?", expected_output="RAG is...", difficulty="medium", category="ai"),
    ]
    path = str(tmp_path / "test.jsonl")
    result_path = export_jsonl(samples, path, format="openai")
    with open(result_path) as f:
        line = json.loads(f.readline())
    assert "messages" in line
    assert line["messages"][0]["role"] == "user"
    assert line["messages"][1]["role"] == "assistant"


def test_export_jsonl_alpaca_format(tmp_path):
    samples = [
        EvalSample(id="1", input="Q?", expected_output="A.", difficulty="easy", category="x"),
    ]
    path = str(tmp_path / "alpaca.jsonl")
    export_jsonl(samples, path, format="alpaca")
    with open(path) as f:
        line = json.loads(f.readline())
    assert "instruction" in line
    assert "output" in line


# ── Full Generator Integration Test ───────────────────────────────────────────

def test_generate_dataset_end_to_end(tmp_path):
    with patch("src.strategies.qa_strategy._LLM", False), \
         patch("src.strategies.adversarial_strategy._LLM", False):
        spec = DatasetSpec(
            topic="Vector databases",
            total_samples=10,
            strategies=["qa", "adversarial"],
            output_dir=str(tmp_path),
        )
        result = generate_dataset(spec)

    assert result.total_generated > 0
    assert result.total_after_validation > 0
    assert os.path.exists(result.output_path)
    assert result.quality_report["original_count"] == result.total_generated
