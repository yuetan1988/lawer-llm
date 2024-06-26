"""
相比rag, 提供更多交互内容, 上传、引用等
"""

from typing import Any, Optional
import os
import shutil
import gradio as gr
import torch
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_texts(file_list):
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_list):
        file_type = one_file.split(".")[-1]
        if file_type == "md":
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == "txt":
            loader = UnstructuredFileLoader(one_file)
        elif file_type == "pdf":
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())

    print(f" length of docs {len(docs)}")
    return docs


def prepare_retrieval_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_retrieval_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def build_index(chunks, embeddings, persist_directory="data_base/vector_db/chroma"):
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()


def prepare_vector_index():
    docs = get_texts(["./2006.15720.pdf"])
    chunks = prepare_retrieval_data(docs)
    embeddings = get_retrieval_model()
    build_index(chunks, embeddings, persist_directory="data_base/vector_db/chroma")


def get_passage_from_query():
    pass


def upload_file(file):
    """用户上传"""
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    application.source_service.add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


class InternLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            .to(torch.bfloat16)
            .cuda()
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


def get_prompt():
    template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
    提供的上下文：
    ···
    {context}
    ···
    用户的问题: {question}
    你给的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    return QA_CHAIN_PROMPT


def load_chain():
    embeddings = get_retrieval_model()

    persist_directory = "data_base/vector_db/chroma"
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings,
    )

    llm = InternLLM(model_name_or_path="/root/share/model_repos/internlm2-chat-7b")
    QA_CHAIN_PROMPT = get_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="mmr"),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa_chain


class Model_center:
    """
    存储问答 Chain 的对象
    """

    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        search = []
        if question == None or len(question) < 1:
            return "", chat_history, search
        try:
            chat_history.append((question, self.chain({"query": question})["result"]))
            print(chat_history)
            return "", chat_history, search
        except Exception as e:
            return e, chat_history, search


def clear_session():
    return "", None, ""


def main():
    model_center = Model_center()

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown(
                    """<h1><center>LayerLLM powered by InternLM</center></h1>
                    <center>书生浦语</center>
                    """
                )
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=1):
                embedding_model = gr.Dropdown(
                    ["text2vec-base", "B3E"], label="向量模型", value="text2vec-base"
                )
                chat = gr.Dropdown(["InternLM"], label="大语言模型", value="InternLM")
                top_k = gr.Slider(
                    1, 20, value=4, step=1, label="检索topk文档", interactive=True
                )
                set_kg_btn = gr.Button("加载知识库")
                file = gr.File(
                    label="上传文件到知识库",
                    visible=True,
                    file_types=[".txt", ".md", ".docx", ".pdf"],
                )

            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(height=400, show_copy_button=True)
                with gr.Row():
                    # 创建一个文本框组件，用于输入 prompt。
                    message = gr.Textbox(label="请输入问题")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot], value="清除历史")
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("生成对话")
                with gr.Row():
                    # 召回参考文献
                    search = gr.Textbox(label="引用文献", max_lines=10)

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            file.upload(upload_file, inputs=file, outputs=None)

            # 点击chat按钮
            db_wo_his_btn.click(
                model_center.qa_chain_self_answer,
                inputs=[message, chatbot],
                outputs=[message, chatbot, search],
            )
            # 输入框回车
            message.submit(
                model_center.qa_chain_self_answer,
                inputs=[message, chatbot],
                outputs=[message, chatbot, search],
            )

        gr.Markdown(
            """提醒：<br>
        1. 本页面仅仅是个demo，不对结果输出负任何法律责任。 <br>
        """
        )
    # threads to consume the request
    gr.close_all()
    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=int(os.environ['PORT1']))
    # 直接启动
    demo.launch()


if __name__ == "__main__":

    # prepare_vector_index()

    main()
