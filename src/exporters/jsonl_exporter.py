"""JSONL exporter — produces fine-tuning compatible JSONL datasets."""
from __future__ import annotations
import json
import os
from typing import List

from src.strategies.base import EvalSample


def export_jsonl(
    samples: List[EvalSample],
    output_path: str,
    format: str = "openai",
) -> str:
    """
    Export samples to JSONL format.

    Args:
        samples: List of EvalSample objects.
        output_path: Destination file path.
        format: 'openai' (messages format), 'alpaca', or 'raw'.

    Returns:
        Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            if format == "openai":
                record = {
                    "messages": [
                        {"role": "user", "content": sample.input},
                        {"role": "assistant", "content": sample.expected_output},
                    ],
                    "metadata": {**sample.metadata, "id": sample.id,
                                 "difficulty": sample.difficulty, "category": sample.category},
                }
            elif format == "alpaca":
                record = {
                    "instruction": sample.input,
                    "input": "",
                    "output": sample.expected_output,
                    "id": sample.id,
                    "difficulty": sample.difficulty,
                    "category": sample.category,
                }
            else:  # raw
                record = {
                    "id": sample.id,
                    "input": sample.input,
                    "expected_output": sample.expected_output,
                    "difficulty": sample.difficulty,
                    "category": sample.category,
                    "metadata": sample.metadata,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return os.path.abspath(output_path)
