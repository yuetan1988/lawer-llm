"""
相比rag3, 使用retrievals开发, 并支持多路召回, rerank等

"""


import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain.llms.base import LLM
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer



logger = logging.getLogger(__name__)



class LangchainLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_tokens: int = 10000
    temperature: float = 0.1
    top_p: float = 0.9
    history: List[str] = []

    def __init__(self, model_name_or_path: str, trust_remote_code: bool = True, **kwargs: Any):
        super(LangchainLLM, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).half().cuda()
        )
        self.model = self.model.eval()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        if hasattr(self.model, 'chat') and callable(self.model.chat):
            response, _ = self.model.chat(self.tokenizer, prompt, history=self.history)
            self.history = self.history + [[None, response]]
            return response
        else:
            batch_inputs = self.tokenizer.batch_encode_plus([prompt], padding='longest', return_tensors='pt')
            batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
            batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()
            
            output = self.model.generate(
                **batch_inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            response = self.tokenizer.batch_decode(
                output.cpu()[:, batch_inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )[0]

            self.history = self.history + [[None, response]]
            return response

    @property
    def _llm_type(self):
        return "Open-retrievals-llm"




if __name__ == "__main__":
    llm = LangchainLLM(model_name_or_path="/root/share/new_models/microsoft/Phi-3-mini-128k-instruct", temperature=0.3)
    response = llm._call("你好")
    print(response)
    print(llm.temperature)
