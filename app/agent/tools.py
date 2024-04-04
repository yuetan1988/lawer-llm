from pydantic import BaseModel


class BaseTool(BaseModel):
    name: str
    description: str

    def use(self, input_text: str) -> str:
        raise NotImplementedError("use() method not implemented")
