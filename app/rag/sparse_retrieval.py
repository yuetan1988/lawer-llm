from langchain_community.retrievers import BM25Retriever
from utils import QueryTracker


class SparseRetrieval:
    def __init__(self, doc_list, metadata=None):
        search_kwargs = {"score_threshold": 0.15, "k": 30}
        self.bm25_retriever = BM25Retriever.from_texts(doc_list, metadata)
        self.bm25_retriever.k = 2

    def query(
        self,
        question: str,
        context_max_length: int = 16000,
        tracker: QueryTracker = None,
    ):
        docs = self.bm25_retriever.get_relevant_documents(question)
        return docs


if __name__ == "__main__":
    from utils import pretty_print_docs

    retrieval = SparseRetrieval(["foo", "bar", "world", "hello", "foo bar"])
    docs = retrieval.query(question="foo ?")
    pretty_print_docs(docs)
