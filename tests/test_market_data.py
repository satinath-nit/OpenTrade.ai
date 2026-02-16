from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from opentrade_ai.data.market_data import MarketDataProvider


class TestMarketDataProvider:
    def setup_method(self):
        self.provider = MarketDataProvider()

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_historical_data_success(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame(
            {
                "Open": range(30),
                "High": range(30),
                "Low": range(30),
                "Close": range(30),
                "Volume": range(30),
            },
            index=dates,
        )
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_historical_data("AAPL", 90, "2025-02-15")
        assert len(result) == 30
        assert "Close" in result.columns

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_historical_data_empty(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(ValueError, match="No data found"):
            self.provider.get_historical_data("INVALID")

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_stock_info(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 28.5,
            "currentPrice": 185.0,
        }
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_stock_info("AAPL")
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 3_000_000_000_000
        assert result["pe_ratio"] == 28.5
        assert result["current_price"] == 185.0

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_stock_info_missing_fields(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_stock_info("AAPL")
        assert result["name"] == "AAPL"
        assert result["sector"] == "Unknown"
        assert result["pe_ratio"] is None

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_recent_news(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "content": {
                    "title": "Apple beats earnings",
                    "provider": {"displayName": "Reuters"},
                    "pubDate": "2025-01-15",
                    "summary": "Strong quarter",
                }
            },
        ]
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_recent_news("AAPL")
        assert len(result) == 1
        assert result[0]["title"] == "Apple beats earnings"
        assert result[0]["publisher"] == "Reuters"

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_recent_news_empty(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.news = None
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_recent_news("AAPL")
        assert result == []

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_financials(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.financials = pd.DataFrame({"Revenue": [100, 200]})
        mock_ticker.balance_sheet = pd.DataFrame({"Assets": [500, 600]})
        mock_ticker.cashflow = pd.DataFrame({"Cash": [50, 60]})
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_financials("AAPL")
        assert "income_statement" in result
        assert "balance_sheet" in result
        assert "cash_flow" in result

    @patch("opentrade_ai.data.market_data.yf.Ticker")
    def test_get_financials_empty(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = self.provider.get_financials("AAPL")
        assert result == {}
