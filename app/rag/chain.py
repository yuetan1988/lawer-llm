from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
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
from rag.llm import InternLLM
from rag.prompt import RAG_PROMPT


def get_prompt_chain(template):
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    return QA_CHAIN_PROMPT


def get_retrieval():
    return


def load_chain(CFG):
    embeddings = HuggingFaceEmbeddings(model_name=CFG.embed_model_name_or_path)

    persist_directory = CFG.vector_db_path
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings,
    )

    llm = InternLLM(model_name_or_path=CFG.llm_model_name_or_path)
    QA_CHAIN_PROMPT = get_prompt_chain(RAG_PROMPT)

    retriever = vectordb.as_retriever(search_type="mmr")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    qa_chain.return_source_documents = True
    return qa_chain


def parse_reference(response):
    reference_text = ""
    for idx, source in enumerate(response["source_documents"][:4]):
        sep = f"参考文献【{idx + 1}】：{source.metadata['header1']}"
        reference_text += f"{sep}\n{source.page_content}\n\n"
    return reference_text


class ModelCenter:
    """
    存储问答 Chain 的对象
    """

    def __init__(self, CFG):
        self.chain = load_chain(CFG)

    def qa_chain_self_answer(self, query: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        reference = []
        if query == None or len(query) < 1:
            return "", chat_history, reference
        try:
            result = self.chain({"query": query})
            chat_history.append((query, result["result"]))
            reference = parse_reference(result)
            return "", chat_history, reference
        except Exception as e:
            return e, chat_history, reference
