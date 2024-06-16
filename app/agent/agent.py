from app.configs.prompt import PromptCN
from app.chat.llm import lmdeploy_response, InternLLM


knowledges = [
    {
        "name_for_human": "法律原文",
        "name_for_model": "法律文件原文",
        "description_for_model": "法律文件原文在判断法律事件时，可以进行查阅具体条款，结合条款与问题帮助回答。",
        "parameters": [
            {
                "name": "法律原文",
                "description": "检索关键词或短语",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
    }
]

Know_DESC = """{name_for_model}: 使用这些知识辅助回答问题. {name_for_human} 知识的用途是什么? {description_for_model} 参数: {parameters} 回答参数为 JSON 对象."""


question = "如果公司没有给我交够全额社保，我应该怎么办?"


def build_system_input():
    knowledge_descs, knowledge_names = [], []
    for knowledge in knowledges:
        knowledge_descs.append(Know_DESC.format(**knowledge))
        knowledge_names.append(knowledge["name_for_model"])

    knowledge_descs = "\n\n".join(knowledge_descs)
    knowledge_names = ",".join(knowledge_names)
    sys_prompt = PromptCN.system_prompt.format(
        knowledge_descs=knowledge_descs, knowledge_names=knowledge_names
    )
    return sys_prompt


sys_prompt = build_system_input()

print(sys_prompt)

llm = InternLLM(model_name_or_path="/root/share/model_repos/internlm2-chat-7b")
response, _ = llm.chat(question, history=[], meta_instruction=sys_prompt)
print(response)
