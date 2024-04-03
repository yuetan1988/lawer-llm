from pydantic import BaseModel


class Agent(BaseModel):
    llm: ChatLLM
    tools: list
    prompt_template: str
    max_loops: int = 10

    @property
    def tool_description(self):
        return

    @property
    def tool_names(self):
        return

    def run(self, question: str):
        history = []
        num_loops = 0

        prompt = self.prompt_template.format(
            tool_description=self.tool_description,
            tool_name=self.tool_name,
            question=question,
            history="{history}",
        )
        while num_loops < self.max_loops:
            num_loops += 1
            cur_prompy = prompt.format()