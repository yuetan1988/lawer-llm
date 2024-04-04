import requests
from bs4 import BeautifulSoup


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
