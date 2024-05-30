from typing import Dict, List, Optional
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity


class CFG:
    base_dir = "/root/lawer-llm/"
    folder_path = base_dir + "inputs/LawBench/data/zero_shot/"
    output_path = base_dir + "outputs/finetune-eval/custom_internlm2"

    # model_name_or_path = output_path + "checkpoint-4000"  # 微调模型
    # model_name_or_path = "/root/share/model_repos/internlm2-chat-7b-sft"  # 原模型
    model_name_or_path = "/root/lawer-llm/outputs/internlm-sft-7b-lora"

    batch_size = 8


if not os.path.exists(CFG.output_path):
    os.makedirs(CFG.output_path)


tokenizer = AutoTokenizer.from_pretrained(
    "/root/share/model_repos/internlm2-chat-7b",
    trust_remote_code=True,
    padding_size="left",
)

model = AutoModelForCausalLM.from_pretrained(
    CFG.model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = model.eval()


def generate_input(item):
    return f"<|User|>:{item['instruction']}\n{item['question']}<eoh>\n<|Bot|>:"


def batch_generate(
    text_inputs: List[str],
    model,
    tokenizer,
    use_sft_adapter: bool = False,
    temp: float = 0.0,
):
    # text_input_format = [PROMPT.format_map({"instruction": input}) for input in text_inputs]
    text_input_format = [generate_input(i) for i in text_inputs]

    batch_inputs = tokenizer.batch_encode_plus(
        text_input_format, padding="longest", return_tensors="pt"
    )
    batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
    batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()

    outputs = model.generate(
        **batch_inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=temp,
        top_p=0.8,
    )
    outputs = outputs.cpu()[:, batch_inputs["input_ids"].shape[-1] :]
    outputs = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )
    return outputs


for filename in os.listdir(CFG.folder_path):
    file_path = os.path.join(CFG.folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        if os.path.exists(f"{CFG.output_path}/{filename}"):
            print(f"SKIP {CFG.output_path}/{filename}")
            continue

        # print(f"start generate: {filename}")
        results = {}
        for i in range(0, len(data), CFG.batch_size):
            text_inputs = [item for item in data[i : i + CFG.batch_size]]
            response = batch_generate(
                text_inputs,
                model,
                tokenizer,
            )
            # response, _ = model.chat(tokenizer, input_text[:1024], history=[])

            for j, item in enumerate(text_inputs):

                curr = {
                    "origin_prompt": text_inputs[j],
                    "prediction": response[j],
                    "refr": f"{item['answer']}",
                }
                results[str(i)] = curr

        with open(f"{CFG.output_path}/{filename}", "w") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
