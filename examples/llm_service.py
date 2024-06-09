from typing import Union
import argparse
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API argparser")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--port", type=str, default="8000")

    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
    # model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True,temperature = 0.00001,seed=2024) ## 需要测试设置seed对输出的影响
    uvicorn.run(app, host="127.0.0.1", port=int(port), workers=1)
