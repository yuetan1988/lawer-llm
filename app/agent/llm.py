from pydantic import BaseModel


class ChatLLM(BaseModel):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0

    def generate(self, prompt: str, stop: List[str] = None):
        return
