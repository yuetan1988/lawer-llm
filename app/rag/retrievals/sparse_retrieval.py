from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import jieba
from rank_bm25 import BM25Okapi
from utils import QueryTracker


class SparseRetrieval:
    def __init__(self, doc_list, metadata=None):
        search_kwargs = {"score_threshold": 0.15, "k": 30}
        self.bm25_retriever = BM25Retriever.from_texts(doc_list, metadata)
        # self.bm25_retriever = BM25Retriever.from_documents()
        self.bm25_retriever.k = 2

    def query(
        self,
        question: str,
        context_max_length: int = 16000,
        tracker: QueryTracker = None,
    ):
        docs = self.bm25_retriever.get_relevant_documents(question)
        return docs


class BM25Model:
    def __init__(self, data_list):
        tokenized_documents = [jieba.lcut(doc) for doc in data_list]
        self.bm25 = BM25Okapi(tokenized_documents)
        self.data_list = data_list

    def query(self, query, k=10):
        query = jieba.lcut(query)  # document和query采用同样的分词
        res = self.bm25.get_top_n(query, self.data_list, n=k)
        return res


def key_words_match_knowledge(dic_all, choices, query):
    result_title = process.extract(query, choices, limit=1)
    match = re.search(r"第([一二三四五六七八九零百十]+)条", query)
    if match:
        result_num = match.group(1)
        key0 = result_title[0][0] + " 第" + result_num + "条"
        if key0 in dic_all.keys():
            return (key0, dic_all[key0])
    return


if __name__ == "__main__":
    from utils import pretty_print_docs

    retrieval = SparseRetrieval(["foo", "bar", "world", "hello", "foo bar"])
    docs = retrieval.query(question="foo ?")
    pretty_print_docs(docs)