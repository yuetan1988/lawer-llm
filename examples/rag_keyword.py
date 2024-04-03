import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "针对如下文档: "
    "### Instruction:\n{instruction}\n\n"
    "从文档中抽取5个关键字"
)


model_name_or_path = "../models"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = model.eval()


print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

messages = []

while True:
    input_text = input("User  >>> ")
    input_text = PROMPT.format_map({"instruction": input_text.replace(" ", "")})
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
