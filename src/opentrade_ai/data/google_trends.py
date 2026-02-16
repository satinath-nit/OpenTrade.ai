import logging

from pytrends.request import TrendReq

logger = logging.getLogger(__name__)


class GoogleTrendsProvider:
    def __init__(self, timeframe: str = "today 3-m"):
        self._timeframe = timeframe

    def get_interest(
        self, ticker: str, company_name: str | None = None
    ) -> dict:
        keyword = f"{company_name or ticker} stock"
        try:
            client = TrendReq(hl="en-US", tz=360)
            client.build_payload([keyword], timeframe=self._timeframe)
            df = client.interest_over_time()

            if df.empty:
                return {
                    "keyword": keyword,
                    "average_interest": 0,
                    "current_interest": 0,
                    "trend": "no_data",
                    "data_points": [],
                    "source": "google_trends",
                }

            col = df.columns[0]
            series = df[col]
            avg = float(series.mean())
            current = float(series.iloc[-1])
            first_half = float(series.iloc[: len(series) // 2].mean()) if len(series) > 1 else avg
            second_half = float(series.iloc[len(series) // 2 :].mean()) if len(series) > 1 else avg

            if second_half > first_half * 1.15:
                trend = "rising"
            elif second_half < first_half * 0.85:
                trend = "declining"
            else:
                trend = "stable"

            return {
                "keyword": keyword,
                "average_interest": round(avg, 1),
                "current_interest": round(current, 1),
                "trend": trend,
                "data_points": series.tolist()[-12:],
                "source": "google_trends",
            }
        except Exception:
            logger.debug("Google Trends fetch failed for %s", ticker, exc_info=True)
            return {
                "keyword": keyword,
                "average_interest": 0,
                "current_interest": 0,
                "trend": "error",
                "data_points": [],
                "source": "google_trends",
            }
