"""QA pair generation strategy — creates question/answer pairs from a topic."""
from __future__ import annotations
import os
import uuid
from typing import List

from src.strategies.base import EvalSample, GenerationStrategy

from src.credential_mode import is_live_openai_key

_client = None
_LLM = is_live_openai_key()

def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        except Exception:
            pass
    return _client


_DIFFICULTY_PROMPTS = {
    "easy": "Generate simple factual questions that have clear, unambiguous answers.",
    "medium": "Generate questions requiring understanding of relationships and cause-effect.",
    "hard": "Generate multi-step reasoning questions requiring synthesis of multiple concepts.",
    "adversarial": "Generate trick questions, edge cases, and prompts designed to cause common LLM failure modes (hallucination, overconfidence, incorrect negation).",
}

_QA_PROMPT = """
Topic: {topic}
Difficulty: {difficulty}
Instruction: {instruction}

Generate {n} question-answer pairs. Each pair should be distinct and non-repetitive.

Respond as a JSON array:
[
  {{"question": "...", "answer": "...", "category": "..."}},
  ...
]
"""


class QAStrategy(GenerationStrategy):
    """Generates question-answer evaluation pairs using LLM or templates."""

    def generate(self, topic: str, n_samples: int, difficulty: str = "medium", **kwargs) -> List[EvalSample]:
        if _LLM:
            return self._generate_with_llm(topic, n_samples, difficulty)
        return self._generate_template(topic, n_samples, difficulty)

    def _generate_with_llm(self, topic: str, n: int, difficulty: str) -> List[EvalSample]:
        instruction = _DIFFICULTY_PROMPTS.get(difficulty, _DIFFICULTY_PROMPTS["medium"])
        prompt = _QA_PROMPT.format(topic=topic, difficulty=difficulty, instruction=instruction, n=n)

        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.8,
        )
        import json, re
        raw = response.choices[0].message.content
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return self._generate_template(topic, n, difficulty)

        pairs = json.loads(match.group())
        samples = []
        for p in pairs[:n]:
            samples.append(EvalSample(
                id=str(uuid.uuid4()),
                input=p.get("question", ""),
                expected_output=p.get("answer", ""),
                difficulty=difficulty,
                category=p.get("category", topic),
                metadata={"strategy": "qa", "source": "llm"},
            ))
        return samples

    def _generate_template(self, topic: str, n: int, difficulty: str) -> List[EvalSample]:
        """Rule-based fallback without LLM API."""
        templates = {
            "easy": [
                ("What is {topic}?", "A foundational concept in {topic} involves..."),
                ("Define {topic} in simple terms.", "{topic} refers to..."),
                ("Name three key components of {topic}.", "The three key components are..."),
            ],
            "medium": [
                ("How does {topic} compare to traditional approaches?", "{topic} differs by..."),
                ("What problem does {topic} solve?", "{topic} addresses the challenge of..."),
                ("Explain the trade-offs involved in {topic}.", "Key trade-offs include..."),
            ],
            "hard": [
                ("Design a system using {topic} for high availability.", "An HA design would involve..."),
                ("What are failure modes in {topic}?", "Common failure modes include..."),
                ("How would you debug a {topic} issue in production?", "Debugging steps: 1..."),
            ],
            "adversarial": [
                ("Is {topic} always the best solution?", "No, {topic} has limitations..."),
                ("Can {topic} replace all traditional methods?", "No, because..."),
                ("{topic} has no disadvantages. True or False?", "False. Key disadvantages are..."),
            ],
        }
        base = templates.get(difficulty, templates["medium"])
        samples = []
        for i in range(n):
            t = base[i % len(base)]
            samples.append(EvalSample(
                id=str(uuid.uuid4()),
                input=t[0].replace("{topic}", topic),
                expected_output=t[1].replace("{topic}", topic),
                difficulty=difficulty,
                category=topic,
                metadata={"strategy": "qa", "source": "template"},
            ))
        return samples
