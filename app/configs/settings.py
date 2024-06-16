import os
from typing import Literal
from pydantic_settings import BaseSettings

from app.configs.constants import APP_NAME

OPENXLAB = True


class Settings(BaseSettings):
    debug: bool = False
    timezone: str = "Asia/Shanghai"

    log_level: Literal["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"] = (
        "INFO"
    )

    # llm_model_path: str = "/root/share/model_repos/internlm2-chat-7b"
    llm_model_path: str = "/root/lawer-llm/outputs/internlm-sft-7b-lora"
    llm_url: str = "http://127.0.0.1:8001/chat"
    knowledge_file_path: str = "/root/lawer-llm/data/download_data/knowledge_laws"
    vector_db_path: str = "/root/lawer-llm/data/database/chroma"
    embed_model_name_or_path: str = "BAAI/bge-large-zh-v1.5"


settings = Settings()

if OPENXLAB:
    print(os.system("pwd"))
    settings.llm_model_path = "./internlm2-weights"
    settings.vector_db_path = "./internlm2-weights/database/chroma"


def make_settings() -> Settings:
    return settings
