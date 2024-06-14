from typing import Any, Optional, List, Union, Tuple

import torch
from langchain.llms.base import LLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.configs.settings import settings


class InternLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            .to(torch.bfloat16)
            .cuda()
        )
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any):
        system_prompt = """You are a helpful, honest, and harmless AI assistant whose name is InternLM (书生·浦语)."""
        messages = [(system_prompt, "")]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"

    def chat(self, prompt, history):
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response, []


def lmdeploy_response(prompt: Union[str, List[str]]):
    pipe = lmdeploy.pipeline(settings.llm_model_path)
    response = pipe(prompt)
    print(response)


if __name__ == "__main__":
    # llm = InternLLM(model_name_or_path=settings.llm_model_path)
    # response = llm._call("回答关于劳动法的案例")
    # print(response)

    lmdeploy_response(["关于劳动法的案例"], settings)
