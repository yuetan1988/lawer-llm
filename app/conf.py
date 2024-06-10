class ErrorInfo:
    """错误与日志类"""

    pass


class PromptCN:
    """指令"""

    # proprocess

    # rag
    rag_prompt = "作为一个精通法律的专家, 请仔细阅读参考材料并使用中文回答问题。记住，如果你不清楚法案直接说不了解，不要试图捏造答案；我们深呼吸一下，一步一步的对问题和资料进行思考。\n 材料：“{}”\n 问题：“{}” "


class Config:
    """超参数与配置类"""

    LLM_DIR = "../models"

    USING_LMDEPLOY = True
    GENERATION_CONFIG_PATH = ""

    USING_RAG = True
    RAG_CONFIG_PATH = ""
    ORIGINAL_KNOWLEDGE_PATH = "/root/lawer-llm/data/download_data/knowledge_laws"
    VECTOR_DB_PATH = "../examples/database/chroma"
    embed_model_name_or_path = "BAAI/bge-large-zh-v1.5"
