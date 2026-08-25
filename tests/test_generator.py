import os
import json
from src.generator import DatasetGenerator

def test_extract_chunks():
    doc = "word " * 150
    gen = DatasetGenerator(doc)
    chunks = gen.extract_chunks(chunk_size=100)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 100
    assert len(chunks[1].split()) == 50

def test_generate_qa_pair():
    gen = DatasetGenerator("")
    pair = gen.generate_qa_pair("Test chunk of text")
    assert "prompt" in pair
    assert "completion" in pair
    assert "Test chunk" in pair["prompt"]

def test_export_jsonl(tmp_path):
    gen = DatasetGenerator("")
    dataset = [{"prompt": "A", "completion": "B"}, {"prompt": "C", "completion": "D"}]
    out_file = os.path.join(tmp_path, "out.jsonl")
    gen.export_jsonl(dataset, out_file)
    
    assert os.path.exists(out_file)
    with open(out_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["prompt"] == "A"
