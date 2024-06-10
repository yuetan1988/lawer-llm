import sys
import logging
from typing import Optional, List, Iterable
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)

from langchain_community.vectorstores import Chroma
from llm import InternLLM
from prompt import RAG_PROMPT
from parse_file import FileParser

logger = logging.getLogger(__name__)


def get_prompt_chain(template):
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    return QA_CHAIN_PROMPT


def get_retrieval():
    return


def get_vector_db(CFG):
    embeddings = HuggingFaceEmbeddings(model_name=CFG.embed_model_name_or_path)

    vector_db = Chroma(
        persist_directory=CFG.VECTOR_DB_PATH,
        embedding_function=embeddings,
    )
    return vector_db


def load_chain(CFG):
    vector_db = get_vector_db(CFG)
    retriever = vector_db.as_retriever(search_type="mmr")

    llm = InternLLM(model_name_or_path=CFG.llm_model_name_or_path)

    QA_CHAIN_PROMPT = get_prompt_chain(RAG_PROMPT)

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


class KnowledgeCenter:
    """
    本地知识
    """

    def __init__(self, CFG, loader, splitter, embedder):
        self.CFG = CFG
        self.loader = loader
        self.splitter = splitter
        self.embedder = embedder

        self.parser = FileParser()
        # self.vector_db = get_vector_db(CFG)

    def init_vector_db(self, file_path: str):
        logger.info(f"Init knowledge from {file_path}")
        document = self.loader(file_path).load_and_split(self.splitter)
        vector_db = Chroma.from_documents(
            documents=doc,
            embedding=self.embedder,
            persist_directory=self.CFG.persist_directory,
        )
        vector_db.persist()

    def add_document(self, file_path: str):
        doc = self.parser(file_path)
        self.vector_db.add_documents(doc)
        self.vector_db.save_local(self.CFG.vector_db_path)


class LawDirectoryLoader(DirectoryLoader):
    def __init__(self, path: str):
        glob = "**/*.md"
        super().__init__(path, loader_cls=TextLoader, glob=glob, show_progress=True)


class LawRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    # https://github.com/chatchat-space/Langchain-Chatchat/blob/master/text_splitter/chinese_recursive_text_splitter.py
    def __init__(
        self,
        separators: Optional[List[str]] = [r"第\S*条 "],
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs,
    ):
        if not separators:
            separators = [
                "\n\n",
                "\n",
                "。|！|？",
                "\.\s|\!\s|\?\s",
                "；|;\s",
                "，|,\s",
            ]

        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        super().__init__(
            separators=separators, is_separator_regex=is_separator_regex, **kwargs
        )

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts = []
        metadatas = []
        for doc in documents:
            md_docs = self.md_splitter.split_text(doc.page_content)
            for md_doc in md_docs:
                texts.append(md_doc.page_content)

                metadatas.append(
                    md_doc.metadata
                    | doc.metadata
                    | {"book": md_doc.metadata.get("header1")}
                )

        return self.create_documents(texts, metadatas=metadatas)

    # def _split_text(self, text: str, separators: List[str]) -> List[str]:
    #     """Split incoming text and return chunks."""
    #     final_chunks = []
    #     # Get appropriate separator to use
    #     separator = separators[-1]
    #     new_separators = []
    #     for i, _s in enumerate(separators):
    #         _separator = _s if self._is_separator_regex else re.escape(_s)
    #         if _s == "":
    #             separator = _s
    #             break
    #         if re.search(_separator, text):
    #             separator = _s
    #             new_separators = separators[i + 1 :]
    #             break

    #     _separator = separator if self._is_separator_regex else re.escape(separator)
    #     splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

    #     # Now go merging things, recursively splitting longer texts.
    #     _good_splits = []
    #     _separator = "" if self._keep_separator else separator
    #     for s in splits:
    #         if self._length_function(s) < self._chunk_size:
    #             _good_splits.append(s)
    #         else:
    #             if _good_splits:
    #                 merged_text = self._merge_splits(_good_splits, _separator)
    #                 final_chunks.extend(merged_text)
    #                 _good_splits = []
    #             if not new_separators:
    #                 final_chunks.append(s)
    #             else:
    #                 other_info = self._split_text(s, new_separators)
    #                 final_chunks.extend(other_info)
    #     if _good_splits:
    #         merged_text = self._merge_splits(_good_splits, _separator)
    #         final_chunks.extend(merged_text)
    #     return [
    #         re.sub(r"\n{2,}", "\n", chunk.strip())
    #         for chunk in final_chunks
    #         if chunk.strip() != ""
    #     ]


if __name__ == "__main__":
    sys.path.append("../")
    from conf import Config

    CFG = Config
    loader = LawDirectoryLoader
    splitter = LawRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=32
    )
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )
    knowledge_center = KnowledgeCenter(
        CFG, loader=loader, splitter=splitter, embedder=embedder
    )
    knowledge_center.init_vector_db(CFG.ORIGINAL_KNOWLEDGE_PATH)
