"""FastAPI entrypoint for the Automated Eval Dataset Generator."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn

from src.generator import DatasetSpec, GenerationResult, generate_dataset

app = FastAPI(
    title="Automated Eval Dataset Generator",
    description="Generate diverse, high-quality evaluation datasets for LLM applications.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Topic or task domain for the dataset")
    total_samples: int = Field(50, ge=5, le=500)
    strategies: List[str] = Field(["qa", "adversarial"])
    difficulty_split: Optional[Dict[str, float]] = None
    export_format: str = Field("openai", pattern="^(openai|alpaca|raw)$")


class GenerateResponse(BaseModel):
    dataset_id: str
    topic: str
    total_generated: int
    total_after_validation: int
    output_path: str
    quality_report: dict


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an evaluation dataset for the given topic and spec."""
    spec = DatasetSpec(
        topic=req.topic,
        total_samples=req.total_samples,
        strategies=req.strategies,
        export_format=req.export_format,
    )
    if req.difficulty_split:
        if abs(sum(req.difficulty_split.values()) - 1.0) > 0.01:
            raise HTTPException(400, "difficulty_split values must sum to 1.0")
        spec.difficulty_split = req.difficulty_split

    result = generate_dataset(spec)
    return GenerateResponse(
        dataset_id=result.dataset_id,
        topic=result.topic,
        total_generated=result.total_generated,
        total_after_validation=result.total_after_validation,
        output_path=result.output_path,
        quality_report=result.quality_report,
    )


@app.get("/health")
def health():
    import os
    return {"status": "ok", "llm_available": bool(os.getenv("OPENAI_API_KEY"))}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
