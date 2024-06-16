import os
import logging
import gradio as gr
import requests
from functools import partial
from requests.adapters import HTTPAdapter
import json
from modelscope import snapshot_download
from app.configs.settings import settings
from app.chat.llm import InternLLM, ImdeployLLM
from app.retrieval.open_retrieval import retrieval, init_index, add_document
from app.configs.prompt import PromptCN

logger = logging.getLogger(__name__)


if not os.path.exists(settings.llm_model_path):
    base_path = "./internlm2-weights"
    os.system(
        f"git clone https://code.openxlab.org.cn/YueTan/lawer-llm.git {base_path}"
    )
    os.system(f"cd {base_path} && git lfs pull")


# llm = InternLLM(model_name_or_path=settings.llm_model_path)
llm = ImdeployLLM(model_name_or_path=settings.llm_model_path)


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


def rewriter_process(prompt, history=[]):
    logging.info(f"Start to rewrite: {prompt}")
    prompt_rewriter = PromptCN.rewriter_prompt.format(instruction=prompt)
    response, history = llm.chat(prompt_rewriter, history)
    logging.info(f"rewrite result: {response}")
    return response


def if_rag_process(prompt, history=[]):
    logging.info(f"Start to judge if rag: {prompt}")
    prompt_retrieval = PromptCN.retrieval_prompt.format(instruction=prompt)
    response, history = llm.chat(prompt_retrieval, history)
    logging.info(f"if rag result: {response}")
    return response


def rag_process(prompt, history=[], top_k: int = 3):
    logging.info(f"Start to rag chat: {prompt}")

    context = retrieval(prompt, top_k=top_k)
    context = [doc.page_content for doc in context]
    logging.info(f"Retrieval result: {context}")

    prompt_with_context = PromptCN.RAG_PROMPT.format(
        question=prompt, context="\n".join(context)
    )

    response, history = chat_process(prompt_with_context, history)
    logging.info(f"chat response: {response}")
    history.append((prompt, response))

    logging.info(f"LLM response: {response}")
    context = parse_reference(context)

    return "", history, context


def app_chat_process(prompt, history=[], top_k: int = 3):
    logging.info(f"Start to chat: {prompt}")
    rewriter_response = rewriter_process(prompt)
    if_rag_response = if_rag_process(rewriter_response)
    if if_rag_response == "1":
        return rag_process(rewriter_response, history)
    else:
        logging.info(f"Normal chat: {prompt}")
        response, history = chat_process(rewriter_response, history, top_k=top_k)
        history.append((prompt, response))
        return "", history, []


def upload_file(file):
    """Handle user file upload and add document to the Chroma database."""
    if not os.path.exists("docs"):
        os.mkdir("docs")

    filename = os.path.basename(file.name)
    file_path = os.path.join("docs", filename)

    with open(file_path, "wb") as f:
        f.write(file)

    add_document(file_path)
    logging.info(f"Uploaded and added document: {filename}")

    return gr.Dropdown.update(choices=[filename], value=filename)


def clear_session():
    return "", None, ""


def parse_reference(reference):
    formatted_text = "\n".join(
        [f"{idx+1}. {item.strip()}" for idx, item in enumerate(reference)]
    )
    return formatted_text


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
                    ["bge-large-zh-v1.5", "bge-base"],
                    label="向量模型",
                    value="bge-large-zh-v1.5",
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
                app_chat_process,
                inputs=[message, chatbot, top_k],
                outputs=[message, chatbot, reference],
            )
            # 输入框回车
            message.submit(
                app_chat_process,
                inputs=[message, chatbot, top_k],
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
