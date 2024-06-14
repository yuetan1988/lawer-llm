from typing import Optional, List, Iterable

from langchain_core.documents import Document
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
