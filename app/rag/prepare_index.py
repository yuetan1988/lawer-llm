from typing import Any, Iterable, List

from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import (
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma


class LawDirectoryLoader(DirectoryLoader):
    def __init__(self, path: str):
        glob = "**/*.md"
        super().__init__(path, loader_cls=TextLoader, glob=glob, show_progress=True)


class LawRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    # https://github.com/chatchat-space/Langchain-Chatchat/blob/master/text_splitter/chinese_recursive_text_splitter.py
    def __init__(self, **kwargs):
        separators = [r"第\S*条 "]
        is_separator_regex = True

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


def prepare_law_index(doc_directory, persist_directory):
    loader = LawDirectoryLoader(doc_directory)
    text_splitter = LawRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=20
    )

    docs = loader.load_and_split(text_splitter=text_splitter)

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding, persist_directory=persist_directory
    )
    vectordb.persist()


def test_langchain_retrieval():
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    persist_directory = "../../examples/database/chroma"

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(search_type="mmr")

    docs = retriever.get_relevant_documents("我是来读宪法的，第一条就是?")
    pretty_print_docs(docs)


if __name__ == "__main__":
    doc_directory = "../../inputs/laws"
    persist_directory = "../../examples/database/chroma"
    prepare_law_index(doc_directory)

    # test_langchain_retrieval()
