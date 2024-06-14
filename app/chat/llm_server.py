from typing import Union
import argparse
import json
import uvicorn
import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from app.configs.settings import settings


app = FastAPI()

model_name_or_path = settings.llm_model_path

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True, device_map="cuda:0"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model = model.eval()
model.generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True, temperature=0.7, seed=42
)


class TranslationRequest(BaseModel):
    input_str: str


def chat(input_str: str):
    global model, tokenizer
    response, history = model.chat(tokenizer, input_str, history=[])
    return response


@app.post("/chat")
async def generate_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    content = json_post_list.get("prompt")

    response = chat(content)
    print(response)

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = {"response": response, "status": 200, "time": time}
    return answer


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API argparser")
    parser.add_argument("--port", type=str, default="8001")

    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=int(args.port), workers=1)
