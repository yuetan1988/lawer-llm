# https://github.com/xxw1995/chatglm3-finetune/blob/main/tools.py


from pydantic import BaseModel


class BaseTool(BaseModel):
    name: str
    description: str

    def use(self, input_text: str) -> str:
        raise NotImplementedError("use() method not implemented")
