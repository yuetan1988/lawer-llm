from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


MODEL_NAME_OR_PATH = "qihoo360/360Zhinao-7B-Chat-4K"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH, device_map="auto", trust_remote_code=True
)

generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME_OR_PATH, trust_remote_code=True
)

messages = []
# round-1
messages.append({"role": "user", "content": "介绍一下刘德华"})
response = model.chat(
    tokenizer=tokenizer, messages=messages, generation_config=generation_config
)
messages.append({"role": "assistant", "content": response})
print(messages)

# round-2
messages.append({"role": "user", "content": "他有什么代表作？"})
response = model.chat(
    tokenizer=tokenizer, messages=messages, generation_config=generation_config
)
messages.append({"role": "assistant", "content": response})
print(messages)
