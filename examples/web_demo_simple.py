import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


model_name_or_path = "/root/share/model_repos/internlm2-chat-7b"
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True, device_map="cuda:0"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model = model.eval()
model.generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True, temperature=0.7, seed=42
)


system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, "")]
response, history = model.chat(tokenizer, messages, history=[])

print(response)
