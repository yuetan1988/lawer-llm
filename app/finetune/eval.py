import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os


#model_name_or_path = "../../outputs/result"
model_name_or_path = '../../models'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()


folder_path = '../../inputs/LawBench/data/zero_shot/'


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        print(f"start generate: {filename}")
        results = {}
        for i, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
            input_text = f"<|User|>:{item['instruction']}\n{item['question']}<eoh>\n<|Bot|>:"

            response, _ = model.chat(tokenizer, input_text)

            answer = f"{item['answer']}"

            curr = {
                "origin_prompt": input_text,
                "prediction": response,
                "refr": answer
            }
            results[str(i)] = curr

        out_path = f'../../outputs/{filename}'
        with open(out_path, 'w') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
