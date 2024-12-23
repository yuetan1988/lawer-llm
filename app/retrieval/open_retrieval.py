import argparse
import logging
import json
from langchain_community.vectorstores import Chroma
from retrievals.tools.langchain import (
    LangchainEmbedding,
    LangchainReranker,
    LangchainLLM,
)
from app.retrieval.file_utils import *
from app.configs.settings import settings

embedder = LangchainEmbedding(model_name="BAAI/bge-large-zh-v1.5")


def load_and_split_documents(file_path: str, chunk_size=256, chunk_overlap=32):
    """
    Helper function to load and split documents from the given file path.
    """
    loader = LawDirectoryLoader(file_path)
    splitter = LawRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return loader.load_and_split(splitter)


def init_index(file_path: str):
    logging.info(f"Init knowledge from {file_path}")
    document = load_and_split_documents(file_path)
    vector_db = Chroma.from_documents(
        documents=document,
        embedding=embedder,
        persist_directory=settings.vector_db_path,
    )
    vector_db.persist()


def add_document(file_path: str):
    documents = load_and_split_documents(file_path)
    vector_db = Chroma(persist_directory=settings.vector_db_path)
    vector_db.add_documents(documents)
    vector_db.persist()


def retrieval(query: str, top_k: int = 3):
    vector_db = Chroma(
        embedding_function=embedder,
        persist_directory=settings.vector_db_path,
    )
    retrieval = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

    context = retrieval.invoke(query)
    return context
