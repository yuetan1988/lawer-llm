from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


base_model_name_or_path = "/root/share/model_repos/internlm2-chat-7b"
lora_model_name_or_path = "/root/lawer-llm/outputs"
merged_output_dir = "/root/lawer-llm/outputs/internlm-sft-7b-lora"
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def batch_generate(
    text_input: List[str],
    model,
    tokenizer,
    use_sft_adapter: bool = True,
    temp: float = 0.7,
):
    text_input_format = [
        PROMPT.format_map({"instruction": input}) for input in text_input
    ]

    batch_inputs = tokenizer.batch_encode_plus(
        text_input_format, padding="longest", return_tensors="pt"
    )
    batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
    batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()

    if use_sft_adapter:
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
        )
    else:
        with model.disable_adapter():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temp,
                top_p=0.8,
            )
    outputs = tokenizer.batch_decode(
        outputs.cpu()[:, batch_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    return outputs


def infer(use_peft: bool = True, save_peft_merge: bool = False):

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True, padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    if use_peft:
        from peft import AutoPeftModelForCausalLM, PeftModel

        model = PeftModel.from_pretrained(model, lora_model_name_or_path)

    model.eval()

    test_input = ["张三怒杀阎婆惜, 该当何罪"]
    outputs = batch_generate(test_input, model, tokenizer, use_sft_adapter=True)
    print(outputs)

    if save_peft_merge:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            merged_output_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        tokenizer.save_pretrained(merged_output_dir)
    return


if __name__ == "__main__":
    infer(use_peft=True, save_peft_merge=True)
