import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from opentrade_ai.data.google_news import GoogleNewsProvider
from opentrade_ai.data.google_trends import GoogleTrendsProvider
from opentrade_ai.data.sec_edgar import SECEdgarProvider

logger = logging.getLogger(__name__)


class MarketDataProvider:
    def __init__(
        self,
        enable_google_news: bool = True,
        enable_sec_edgar: bool = True,
        enable_google_trends: bool = True,
        google_news_period: str = "7d",
        google_news_max_results: int = 10,
        sec_edgar_max_filings: int = 5,
        google_trends_timeframe: str = "today 3-m",
    ):
        self._enable_google_news = enable_google_news
        self._enable_sec_edgar = enable_sec_edgar
        self._enable_google_trends = enable_google_trends

        self._google_news: GoogleNewsProvider | None = None
        self._sec_edgar: SECEdgarProvider | None = None
        self._google_trends: GoogleTrendsProvider | None = None

        if enable_google_news:
            self._google_news = GoogleNewsProvider(
                max_results=google_news_max_results, period=google_news_period
            )
        if enable_sec_edgar:
            self._sec_edgar = SECEdgarProvider(max_filings=sec_edgar_max_filings)
        if enable_google_trends:
            self._google_trends = GoogleTrendsProvider(
                timeframe=google_trends_timeframe
            )

    def get_historical_data(
        self, ticker: str, period_days: int = 90, end_date: str | None = None
    ) -> pd.DataFrame:
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        start = end - timedelta(days=period_days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return df

    def get_stock_info(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", None)),
            "revenue_growth": info.get("revenueGrowth", None),
            "profit_margins": info.get("profitMargins", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "return_on_equity": info.get("returnOnEquity", None),
            "free_cash_flow": info.get("freeCashflow", None),
        }

    def get_recent_news(self, ticker: str) -> list[dict]:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        results = []
        for item in news[:10]:
            content = item.get("content", {})
            results.append({
                "title": content.get("title", item.get("title", "No title")),
                "publisher": content.get("provider", {}).get(
                    "displayName", item.get("publisher", "Unknown")
                ),
                "published": content.get("pubDate", ""),
                "summary": content.get("summary", ""),
            })
        return results

    def get_financials(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        result = {}
        income = stock.financials
        if income is not None and not income.empty:
            result["income_statement"] = income.head(4).to_dict()
        balance = stock.balance_sheet
        if balance is not None and not balance.empty:
            result["balance_sheet"] = balance.head(4).to_dict()
        cashflow = stock.cashflow
        if cashflow is not None and not cashflow.empty:
            result["cash_flow"] = cashflow.head(4).to_dict()
        return result

    def get_google_news(
        self, ticker: str, company_name: str | None = None
    ) -> list[dict]:
        if not self._google_news:
            return []
        return self._google_news.get_news(ticker, company_name)

    def get_sec_filings(
        self, ticker: str, form_types: list[str] | None = None
    ) -> list[dict]:
        if not self._sec_edgar:
            return []
        return self._sec_edgar.get_filings(ticker, form_types)

    def get_google_trends(
        self, ticker: str, company_name: str | None = None
    ) -> dict:
        if not self._google_trends:
            return {
                "keyword": "",
                "average_interest": 0,
                "current_interest": 0,
                "trend": "disabled",
                "data_points": [],
                "source": "google_trends",
            }
        return self._google_trends.get_interest(ticker, company_name)
