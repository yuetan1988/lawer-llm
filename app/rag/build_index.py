from typing import Any, Iterable, List

import tiktoken
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import (
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import spacy
import PyPDF2


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
        **kwargs
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

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]


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


def test_document_spliter():
    loader = LawDirectoryLoader(doc_directory)
    text_splitter = LawRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=20
    )

    docs = loader.load_and_split(text_splitter=text_splitter)
    return


def test_langchain_retrieval():
    from utils import pretty_print_docs

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
