from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, Response, status

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


app = FastAPI()


model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


class GenerateParameters(BaseModel):
    best_of: Optional[int] = 1
    decoder_input_details: bool = False
    details: bool = True
    do_sample: bool = True
    frequency_penalty: Optional[float] = 0
    grammar: Optional[str] = None
    max_new_tokens: int = 200
    repetition_penalty: Optional[float] = 1
    return_full_text: bool = False
    seed: Optional[int] = None
    stop: List[str] = (["\n\n"],)
    temperature: float = 0.5
    top_k: Optional[int] = 10
    top_n_tokens: Optional[int] = 5
    top_p: Optional[float] = 0.95
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: bool


class GenerateRequest(BaseModel):
    inputs: str
    parameters: GenerateParameters


class GenerateResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # Prepare arguments based on request
    generation_args = {
        "max_new_tokens": request.parameters.max_new_tokens,
        "return_full_text": request.parameters.return_full_text,
        "temperature": request.parameters.temperature,
    }

    if request.parameters.do_sample:
        generation_args.update(
            {
                "do_sample": request.parameters.do_sample,
                "top_k": request.parameters.top_k,
            }
        )

    if request.parameters.seed:
        torch.manual_seed(request.parameters.seed)

    try:
        # Generate text from the model
        output = pipe(request.inputs, **generation_args)
        generated_text = output[0]["generated_text"]
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
