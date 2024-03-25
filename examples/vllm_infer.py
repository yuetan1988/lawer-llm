
from vllm import LLM
from vllm import SamplingParams

sampling_params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=512)


def infer_by_batch(all_raw_text, llm, system):
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            llm.tokenizer,
            q,
            system=system,
            max_window_size=6144,
            chat_format='chatml',
        )
        batch_raw_text.append(raw_text)
    res = llm.model.generate(batch_raw_text, sampling_params, use_tqdm = False)
    res = [output.outputs[0].text.replace('<|im_end|>', '').replace('\n', '') for output in res]
    return res
