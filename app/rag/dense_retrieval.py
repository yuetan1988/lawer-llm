from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import Chroma as Vectorstore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from utils import QueryTracker
from BCEmbedding.tools.langchain import BCERerank


class DenseRetrieval:
    def __init__(self, persist_directory):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
        vectordb = Vectorstore(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            # allow_dangerous_deserialization=True,
            # distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        search_kwargs = {"score_threshold": 0.15, "k": 30}
        self.retriever = vectordb.as_retriever(search_type="similarity")

        reranker_args = {
            "model": "../../inputs/bce-reranker-base_v1",
            "top_n": 7,
            "device": "cuda",
            "use_fp16": True,
        }
        self.reranker = BCERerank(**reranker_args)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.retriever
        )

    def query(
        self,
        question: str,
        context_max_length: int = 16000,
        tracker: QueryTracker = None,
    ):
        docs = self.compression_retriever.get_relevant_documents(question)
        pretty_print_docs(docs)


class CacheRetrieval:
    def __init__(self):
        pass

    def clear_all(self):
        pass


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


if __name__ == "__main__":
    persist_directory = "../../examples/database/chroma"
    dense_retrieval = DenseRetrieval(persist_directory)
    dense_retrieval.query(question="我是来读宪法的，第一条就是?")
