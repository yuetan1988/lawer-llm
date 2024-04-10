"""
ReAct: 先思考再行动，行动结果返回后再思考下一步该做什么，不断重复
"""

from pydantic import BaseModel
from llm import ChatLLM
from tools import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT = ""


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


if __name__ == "__main__":
    prompt = input("Enter a question / task for the agent: ")
    agent = Agent()
    result = agent.run(prompt)
    print(f"Final answer is {result}")
