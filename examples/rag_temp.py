from typing import Optional, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import gradio as gr


def get_texts(file_list):
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_list):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    
    print(f" length of docs {len(docs)}")
    return docs


def prepare_retrieval_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_retrieval_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def build_index(chunks, embeddings, persist_directory = 'data_base/vector_db/chroma'):    
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()


def prepare_vector_index():
    docs = get_texts(['./2006.15720.pdf'])
    chunks = prepare_retrieval_data(docs)
    embeddings = get_retrieval_model()
    build_index(chunks, embeddings, persist_directory = 'data_base/vector_db/chroma')

