"""
Automated Eval Dataset Generator
=================================
Orchestrates multiple strategies to produce balanced, high-quality
evaluation datasets for LLM applications.
"""
from __future__ import annotations
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.strategies.base import EvalSample, GenerationStrategy
from src.strategies.qa_strategy import QAStrategy
from src.strategies.adversarial_strategy import AdversarialStrategy
from src.exporters.jsonl_exporter import export_jsonl
from src.validators.quality_check import validate_and_deduplicate


_STRATEGIES: Dict[str, GenerationStrategy] = {
    "qa": QAStrategy(),
    "adversarial": AdversarialStrategy(),
}

_DEFAULT_DIFFICULTY_SPLIT = {
    "easy": 0.25,
    "medium": 0.40,
    "hard": 0.25,
    "adversarial": 0.10,
}


@dataclass
class DatasetSpec:
    topic: str
    total_samples: int = 100
    strategies: List[str] = field(default_factory=lambda: ["qa", "adversarial"])
    difficulty_split: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_DIFFICULTY_SPLIT))
    export_format: str = "openai"  # openai | alpaca | raw
    output_dir: str = "datasets"


@dataclass
class GenerationResult:
    dataset_id: str
    topic: str
    samples: List[EvalSample]
    output_path: str
    quality_report: dict
    total_generated: int
    total_after_validation: int


def generate_dataset(spec: DatasetSpec) -> GenerationResult:
    """
    Generate a complete evaluation dataset according to the given spec.
    Applies multiple strategies, validates, deduplicates, and exports.
    """
    dataset_id = str(uuid.uuid4())[:8]
    print(f"\n[generator] Starting dataset generation for: {spec.topic!r}")
    print(f"[generator] Target: {spec.total_samples} samples | Strategies: {spec.strategies}")

    all_samples: List[EvalSample] = []

    # ── Generate from each strategy ───────────────────────────────────────────
    for strategy_name in spec.strategies:
        strategy = _STRATEGIES.get(strategy_name)
        if not strategy:
            print(f"[generator] Unknown strategy: {strategy_name}, skipping")
            continue

        strategy_n = spec.total_samples // len(spec.strategies)

        if strategy_name == "qa":
            # Split by difficulty
            for difficulty, ratio in spec.difficulty_split.items():
                if difficulty == "adversarial":
                    continue  # handled by adversarial strategy
                n = max(1, int(strategy_n * ratio / sum(
                    v for k, v in spec.difficulty_split.items() if k != "adversarial"
                )))
                print(f"  [{strategy_name}] Generating {n} {difficulty} samples...")
                samples = strategy.generate(spec.topic, n_samples=n, difficulty=difficulty)
                all_samples.extend(samples)
        else:
            # Adversarial: generate remaining allocation
            adv_n = max(1, int(spec.total_samples * spec.difficulty_split.get("adversarial", 0.10)))
            print(f"  [{strategy_name}] Generating {adv_n} adversarial samples...")
            samples = strategy.generate(spec.topic, n_samples=adv_n)
            all_samples.extend(samples)

    total_generated = len(all_samples)
    print(f"[generator] Generated {total_generated} raw samples")

    # ── Validate and deduplicate ──────────────────────────────────────────────
    cleaned, quality_report = validate_and_deduplicate(all_samples)
    print(f"[generator] After validation: {len(cleaned)} samples "
          f"({quality_report['removed_duplicates']} duplicates removed, "
          f"{quality_report['removed_too_short']} too short)")

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(spec.output_dir, exist_ok=True)
    slug = spec.topic.lower().replace(" ", "-")[:30]
    output_path = os.path.join(spec.output_dir, f"{dataset_id}_{slug}.jsonl")
    export_jsonl(cleaned, output_path, format=spec.export_format)
    print(f"[generator] Exported to {output_path}")

    return GenerationResult(
        dataset_id=dataset_id,
        topic=spec.topic,
        samples=cleaned,
        output_path=output_path,
        quality_report=quality_report,
        total_generated=total_generated,
        total_after_validation=len(cleaned),
    )
