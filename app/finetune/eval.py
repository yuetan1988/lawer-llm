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
    output_path = base_dir + "outputs/finetune-eval/internlm2-chat-7b-sft"
    # model_name_or_path = output_path + "checkpoint-4000"  # 微调模型
    model_name_or_path = "/root/share/model_repos/internlm2-chat-7b-sft"  # 原模型

    batch_size = 8


if not os.path.exists(CFG.output_path):
    os.makedirs(CFG.output_path)


tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_name_or_path, trust_remote_code=True
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
        for i in range(0, len(data), batch_size):
            text_inputs = [item for item in data[i : i + batch_size]]

            text_input_format = [generate_input(i) for i in text_inputs]
            batch_inputs = tokenizer.batch_encode_plus(
                text_input_format, padding="longest", return_tensors="pt"
            )
            batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
            batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()

            generation_args = {
                "max_new_tokens": 512,
                "temperature": 0.0,
                "do_sample": False,
                "top_p": 0.8,
            }
            generate_ids = model.generate(
                **batch_inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args,
            )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # response, _ = model.chat(tokenizer, input_text[:1024], history=[])

            answer = f"{item['answer']}"

            curr = {"origin_prompt": input_text, "prediction": response, "refr": answer}
            results[str(i)] = curr

        with open(f"{CFG.output_path}/{filename}", "w") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
