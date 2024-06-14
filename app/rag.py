import logging
import gradio as gr
import requests
from functools import partial
from requests.adapters import HTTPAdapter
import json
from app.configs.settings import settings
from app.chat.llm import InternLLM
from app.retrieval.dense_retrieval import retrieval
from app.configs.prompt import PromptCN

logger = logging.getLogger(__name__)

llm = InternLLM(model_name_or_path=settings.llm_model_path)


def chat_process_url(prompt):
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"prompt": prompt})
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=3))
    try:
        res = s.post(settings.llm_url, data=data, headers=headers, timeout=600)
        if res.status_code == 200:
            return res.json()["response"]
        else:
            return None
    except requests.exceptions.RequestException as e:
        logging.warning(e)
        return None


def chat_process(prompt, history):
    response, history = llm.chat(prompt, history)
    return response, history


def rag_process(prompt, history=[]):
    logging.info(f"Start to chat: {prompt}")

    context = retrieval(prompt)
    context = [doc.page_content for doc in context]
    logging.info(f"Retrieval result: {context}")

    prompt_with_context = PromptCN.RAG_PROMPT.format(
        question=prompt, context="\n".join(context)
    )

    response, history = chat_process(prompt_with_context, history)
    print(history)
    history.append((prompt, response))
    print(history)

    logging.info(f"LLM response: {response}")
    context = parse_reference(context)

    return "", history, context


def upload_file(file):
    """用户上传"""
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    # shutil.move(file.name, "docs/" + filename)
    # # file_list首位插入新上传的文件
    # file_list.insert(0, filename)
    # add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file, value=filename)


def clear_session():
    return "", None, ""


def parse_reference(reference):
    return ["\n".join(reference)]


def main():
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown(
                    """<h1><center>LawerLLM powered by InternLM</center></h1>"""
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
                    1, 20, value=3, step=1, label="检索topk文档", interactive=True
                )
                set_kg_btn = gr.Button("加载知识库")
                file = gr.File(
                    label="上传文件到知识库",
                    visible=True,
                    file_types=[".txt", ".md", ".docx", ".pdf"],
                )

            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(height=470, show_copy_button=True)
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
                    reference = gr.Textbox(label="引用文献", max_lines=10)

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            file.upload(
                upload_file,
                inputs=[file],
                outputs=None,
            )

            # 点击chat按钮
            db_wo_his_btn.click(
                rag_process,
                inputs=[message, chatbot],
                outputs=[message, chatbot, reference],
            )
            # 输入框回车
            message.submit(
                rag_process,
                inputs=[message, chatbot],
                outputs=[message, chatbot, reference],
            )

        gr.Markdown(
            """<br>
            本页面是罪恶克星demo, 结果仅供参考, 感谢书生浦语大模型提供的算力与模型支持。 <br>
            """
        )
    # threads to consume the request
    gr.close_all()

    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=int(os.environ['PORT1']))
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    # rag_process("介绍下劳动法")

    main()
