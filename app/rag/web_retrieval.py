import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class WebRetrieval(object):
    def __init__(self, config: dict, retry: int = 2):
        self.config = config
        self.retry = retry

    def fetch_url(self, query, target_link):
        if not target_link.startswith("http"):
            return

        logger.info(f"[web retrieval] extract: {target_link}")

        try:
            response = requests.get(target_link, timeout=30)
        except Exception as e:
            logger.errer("fetch_url {}".format(str(e)))


class DuckduckgoSearch(object):
    def __init__(self):
        pass

    def search(self, query):
        try:
            with DDGS(timeout=20) as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                print(results)

            web_content = ""
            if results:
                for result in results:
                    web_content += result["body"]
            return web_content
        except Exception as e:
            print(f"网络检索异常:{query}")
            return ""


if __name__ == "__main__":
    r = DuckduckgoSearch().search("马保国")
    print(r[:2])
