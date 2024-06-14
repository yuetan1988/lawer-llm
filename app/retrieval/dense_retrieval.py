import argparse
import logging
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.retrieval.file_utils import *
from app.configs.settings import settings


embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": False},
)


def index(file_path: str):
    logging.info(f"Init knowledge from {file_path}")

    loader = LawDirectoryLoader(file_path)
    splitter = LawRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=32
    )
    document = loader.load_and_split(splitter)
    vector_db = Chroma.from_documents(
        documents=document,
        embedding=embedder,
        persist_directory=settings.vector_db_path,
    )
    vector_db.persist()


def retrieval(query: str):
    vector_db = Chroma(
        embedding_function=embedder,
        persist_directory=settings.vector_db_path,
    )
    retrieval = vector_db.as_retriever(search_type="mmr")

    context = retrieval.invoke(query)
    return context


if __name__ == "__main__":

    # index(settings.knowledge_file_path)

    context = retrieval("劳动法")
    print(context)
