"""
相比rag2, 提供更多后端pipe
"""

from collections import defaultdict
from typing import Any, List, Optional

import gradio as gr
import torch
from langchain.chains import LLMChain, RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever, StrOutputParser
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class InternLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any):

        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
                        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
                        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
                        """
        messages = [(system_prompt, "")]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"


multi_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，以从矢量数据库中检索相关文档。
                                通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
                                提供这些用换行符分隔的替代问题，不要给出多余的回答。问题：{question}"""  # noqa
MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    template=multi_query_prompt_template, input_variables=["question"]
)


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def get_retriever(retriever: BaseRetriever, llm):
    retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )
    return retriever


def combine_law_docs(docs: List[Document]) -> str:
    # 将检索到的法条合并为str
    # 相关法律：《中华人民共和国刑法》
    # 第一条 XXXX
    # 相关法律：《中华人民共和国宪法》
    # 第三条 XXXX
    law_books = defaultdict(list)
    for doc in docs:
        metadata = doc.metadata
        if "book" in metadata:
            law_books[metadata["book"]].append(doc)

    law_str = ""
    for book, docs in law_books.items():
        law_str += f"相关法律：《{book}》\n"
        law_str += "\n".join([doc.page_content.strip("\n") for doc in docs])
        law_str += "\n"

    return law_str


def test_law_db():
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    # 向量数据库持久化路径
    persist_directory = "./database"

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    llm = InternLLM(model_name_or_path="../models")

    # Prompt 模板
    template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    回答: 
    """

    prompt = PromptTemplate.from_template(template)

    # retriever = vectordb.as_retriever()
    retriever = get_retriever(vectordb.as_retriever(), llm)

    chain = (
        {"context": retriever | combine_law_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # test
    while 1:
        input_text = input("User  >>> ")
        input_text = input_text.replace(" ", "")
        if input_text == "exit":
            break

        result = chain.invoke(input_text)
        print("检索问答链回答 question 的结果：")
        print(result)


if __name__ == "__main__":
    test_law_db()
