"""
相比rag2, 提供更多后端pipe
"""

from typing import Optional, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import gradio as gr


class InternLLM(LLM):
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path):     
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, trust_remote_code=True).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")
    
    def _call(self, prompt : str, stop = None,
                run_manager = None,
                **kwargs: Any):
     
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
                        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
                        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
                        """
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "InternLM"


def test_law_db():
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    # 向量数据库持久化路径
    persist_directory = './database'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    llm = InternLLM(model_name_or_path='../models')

    # Prompt 模板
    template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    回答: 
    """

    prompt = ChatPromptTemplate.from_template(template)

    # multi_query_retriever
    retriever = get_retriever(retriever = vectordb.as_retriever(), model = llm)

    chain = (
        {"context": retriever | combine_law_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

    # test
    while 1:
        input_text = input("User  >>> ")
        input_text = input_text.replace(' ', '')
        if input_text == "exit":
            break

        result = chain.invoke(input_text)
        print("检索问答链回答 question 的结果：")
        print(result)


if __name__ == '__main__':
    test_law_db()
