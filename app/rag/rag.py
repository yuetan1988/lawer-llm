import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

from llm import InternLLM


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
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    persist_directory = "../../examples/database/chroma"
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings,
    )

    llm = InternLLM(model_name_or_path="../../models")
    QA_CHAIN_PROMPT = get_prompt()

    retriever = vectordb.as_retriever(search_type="mmr")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa_chain


class ModelCenter:
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
            return "", chat_history, search
        except Exception as e:
            return e, chat_history, search


def clear_session():
    return "", None, ""


def main():
    model_center = ModelCenter()

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
                    ["text2vec-base", "bge-base"],
                    label="向量模型",
                    value="text2vec-base",
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
            1. 本页面仅仅是个名为罪恶克星的demo, 不对结果输出负任何法律责任。 <br>
            """
        )
    # threads to consume the request
    gr.close_all()
    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=int(os.environ['PORT1']))
    # 直接启动
    demo.launch()


if __name__ == "__main__":
    main()
