from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class DatasetRequest(BaseModel):
    source_text: str
    num_samples: int = 5

def generate_dataset(req: DatasetRequest) -> dict:
    # Mocking dataset generation from source text using LLM
    samples = []
    for i in range(req.num_samples):
        samples.append({
            "instruction": f"Generate question {i+1} about the text",
            "context": req.source_text,
            "response": f"Mock response for question {i+1}"
        })
    return {"dataset": samples}

@app.post("/generate-dataset")
async def generate(req: DatasetRequest):
    return generate_dataset(req)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
