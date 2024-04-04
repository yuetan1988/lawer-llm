from typing import Any, Optional

import torch
from langchain.llms.base import LLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
                        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
                        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
                        """
        messages = [(system_prompt, "")]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"


if __name__ == "__main__":
    llm = InternLLM(model_name_or_path="../../models")
    response = llm._call("你好")
    print(response)
