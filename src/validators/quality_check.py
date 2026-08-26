"""Quality validation: deduplication, length checks, and coverage analysis."""
from __future__ import annotations
from collections import Counter
from typing import List, Tuple
import re

from src.strategies.base import EvalSample


def validate_and_deduplicate(
    samples: List[EvalSample],
    min_input_len: int = 10,
    min_output_len: int = 5,
    similarity_threshold: float = 0.85,
) -> Tuple[List[EvalSample], dict]:
    """
    Run quality checks on a dataset:
    1. Remove samples that are too short
    2. Remove near-duplicate inputs (jaccard similarity)
    3. Return cleaned samples + a quality report

    Returns: (cleaned_samples, report_dict)
    """
    report = {
        "original_count": len(samples),
        "removed_too_short": 0,
        "removed_duplicates": 0,
        "final_count": 0,
        "difficulty_distribution": {},
        "category_distribution": {},
    }

    # Step 1: Length filter
    length_filtered = []
    for s in samples:
        if len(s.input.strip()) >= min_input_len and len(s.expected_output.strip()) >= min_output_len:
            length_filtered.append(s)
        else:
            report["removed_too_short"] += 1

    # Step 2: Near-duplicate removal (Jaccard on token sets)
    seen: List[EvalSample] = []
    for candidate in length_filtered:
        if not _is_near_duplicate(candidate, seen, threshold=similarity_threshold):
            seen.append(candidate)
        else:
            report["removed_duplicates"] += 1

    # Step 3: Compute distributions
    difficulty_dist = Counter(s.difficulty for s in seen)
    category_dist = Counter(s.category for s in seen)
    report["difficulty_distribution"] = dict(difficulty_dist)
    report["category_distribution"] = dict(category_dist)
    report["final_count"] = len(seen)

    return seen, report


def _tokenize(text: str) -> set:
    """Simple whitespace+punctuation tokenizer."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_near_duplicate(candidate: EvalSample, existing: List[EvalSample], threshold: float) -> bool:
    cand_tokens = _tokenize(candidate.input)
    for e in existing:
        if _jaccard(cand_tokens, _tokenize(e.input)) >= threshold:
            return True
    return False
