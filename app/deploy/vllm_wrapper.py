"""
vllm部署, 主要优化在于: batch, paged attention
"""

import copy

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams


class vLLMWrapper(object):
    def __init__(
        self,
        model_dir,
        tensor_parallel_size=1,
        gpu_mempry_utilization=0.9,
        dtype="float16",
        quantization=None,
    ):
        self.generation_config = GenerationConfig.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        self.stop_words_ids = [
            self.tokenizer.im_start_id,
            self.tokenizer.im_end_id,
            self.tokenizer.eos_token_id,
        ]
        os.environ["VLLM_USE_MODELSCOPE"] = "True"
        self.model = LLM(
            model=model_dir,
            tokenizer=model_dir,
            tensor_parallel_size=tensor_parallel_size,  # tp
            trust_remote_code=True,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,  # 0.6
            dtype=dtype,
        )

    def generate(self, query, history=None, system=None, extra_stop_words_ids=None):
        if isinstance(inputs, str):
            inputs = [inputs]

        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)

        sampling_params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=512)
        response = self.model.generate(
            prompt_token_ids=[prompt_tokens],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        response = [resp.outputs[0].text for resp in response]

        response_token_ids = remove_stop_words(
            req_output.outputs[0].token_ids, stop_words_ids
        )
        response = self.tokenizer.decode(response_token_ids)

        history.append((query, response))
        return response, history


def infer_by_batch(all_raw_text, llm, system):
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            llm.tokenizer,
            q,
            system=system,
            max_window_size=6144,
            chat_format="chatml",
        )
        batch_raw_text.append(raw_text)
    res = llm.model.generate(batch_raw_text, sampling_params, use_tqdm=False)
    res = [
        output.outputs[0].text.replace("<|im_end|>", "").replace("\n", "")
        for output in res
    ]
    return res
