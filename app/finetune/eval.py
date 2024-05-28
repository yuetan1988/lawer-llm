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


if not os.path.exists(CFG.output_path):
    os.makedirs(CFG.output_path)


tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_name_or_path, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.eval()


for filename in os.listdir(CFG.folder_path):
    file_path = os.path.join(CFG.folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        if os.path.exists(f"{CFG.output_path}/{filename}"):
            print(f"SKIP {out_path}/{filename}")
            continue

        # print(f"start generate: {filename}")
        results = {}
        for i, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
            input_text = (
                f"<|User|>:{item['instruction']}\n{item['question']}<eoh>\n<|Bot|>:"
            )

            # generation_args = {
            #     "max_new_tokens": 500,
            #     "temperature": 0.0,
            #     "do_sample": False,
            # }
            # generate_ids = model.generate(inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

            # # remove input tokens
            # generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            response, _ = model.chat(tokenizer, input_text)

            answer = f"{item['answer']}"

            curr = {"origin_prompt": input_text, "prediction": response, "refr": answer}
            results[str(i)] = curr

        with open(f"{CFG.output_path}/{filename}", "w") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
