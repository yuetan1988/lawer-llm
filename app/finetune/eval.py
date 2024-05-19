import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity


class CFG:
    base_dir = ""
    folder_path = base_dir + "../../inputs/LawBench/data/zero_shot/"
    output_path = base_dir + ""
    # model_name_or_path = "../../outputs/result"
    model_name_or_path = "../../models/internlm2-chat-7b"  # 原模型


if not os.path.exists(CFG.output_path):
    os.makedirs(CFG.output_path)


tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_name_or_path, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_name_or_path,
    trust_remote_code=True,
    # load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.eval()


for filename in os.listdir(CFG.folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        if os.path.exists(f"{CFG.output_path}/{filename}"):
            print(f"SKIP {out_path}/{filename}")
            continue

        print(f"start generate: {filename}")
        results = {}
        for i, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
            input_text = (
                f"<|User|>:{item['instruction']}\n{item['question']}<eoh>\n<|Bot|>:"
            )

            response, _ = model.chat(tokenizer, input_text)

            answer = f"{item['answer']}"

            curr = {"origin_prompt": input_text, "prediction": response, "refr": answer}
            results[str(i)] = curr

        with open(f"{CFG.output_path}/{filename}", "w") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
