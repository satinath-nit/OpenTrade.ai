import logging

from gnews import GNews

logger = logging.getLogger(__name__)


class GoogleNewsProvider:
    def __init__(self, max_results: int = 10, period: str = "7d"):
        self._max_results = max_results
        self._period = period

    def get_news(
        self, ticker: str, company_name: str | None = None
    ) -> list[dict]:
        query = f"{company_name or ticker} stock"
        try:
            client = GNews(
                language="en",
                country="US",
                period=self._period,
                max_results=self._max_results,
            )
            raw = client.get_news(query)
            if not raw:
                return []
            results = []
            for item in raw:
                pub = item.get("publisher") or {}
                results.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "published": item.get("published date", ""),
                    "publisher": (
                        pub.get("title", "Unknown")
                        if isinstance(pub, dict)
                        else "Unknown"
                    ),
                    "url": item.get("url", ""),
                    "source": "google_news",
                })
            return results
        except Exception:
            logger.debug("Google News fetch failed for %s", ticker, exc_info=True)
            return []
