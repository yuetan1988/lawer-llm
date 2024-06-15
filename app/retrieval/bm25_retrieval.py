import math
import os
import jieba
import logging
import json
import uuid
from typing import Optional, List


class BM25Searcher:
    def __init__(
        self,
        documents,
        chunk_size=None,
        chunk_overlap=None,
        splitter=None,
        stop_words_dir: Optional[str] = None,
    ):
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self.splitter = splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stop_words_dir = stop_words_dir

        if self.splitter:
            documents = self.splitter.split_text(self.documents)
        else:
            documents = self.documents

        self.bm25 = BM25Okapi(documents)

    def search(self, query: str, top_k: int, batch_size: int = -1) -> List[str]:
        scores = self.bm25.get_scores(query)
        sorted_docs = sorted(
            zip(self.documents, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return sorted_docs

    def _load_stop_words(self):
        stop_words_path = os.path.join(self.stop_words_dir, "stop_words.txt")
        if not os.path.exists(stop_words_path):
            raise Exception(f"system stop words: {stop_words_path} not found")

        stop_words = []
        with open(stop_words_path, "r", encoding="utf8") as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words


if __name__ == "__main__":
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?",
    ]

    retrieval = BM25Searcher([doc.split(" ") for doc in corpus])
    res = retrieval.search("windy London".split(" "), top_k=2)
    print(res)
