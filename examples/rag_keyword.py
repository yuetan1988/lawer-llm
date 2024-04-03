import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

KEYWORD_PROMPT = (
    "从这句话中抽取5个和法律、条例、规定相关的关键字 "
    "### \n{instruction}\n"
    "只输出关键字即可, 不要说多余的话"
)

# 茴香豆参考
TOPIC_PROMPT = "告诉我这句话的主题，直接说主题不要解释" "\n{instruction}\n"

SCORING_QUESTION_PROMPT = (
    "{instruction}\n"
    "请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。"
)

SCORING_RELAVANCE_PROMPT = (
    "问题：{query}\n材料：{passage}\n"
    "请仔细阅读以上内容，判断问题和材料的关联度，用0～10表示。判断标准：非常相关得 10 分；完全没关联得 0 分。直接提供得分不要解释。"
)
SUMMARIZE_PROMPT = "{query} \n" "仔细阅读以上内容，输出其摘要，总结得简短有力"


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
