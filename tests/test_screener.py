from unittest.mock import MagicMock, patch

import pandas as pd

from opentrade_ai.config import AppConfig, LLMConfig, TradingConfig
from opentrade_ai.screener import (
    OpenTradeScreener,
    ScreenerPick,
    ScreenerResult,
    parse_watchlist_input,
)


class TestParseWatchlist:
    def test_parse_commas(self):
        assert parse_watchlist_input("AAPL, MSFT,GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_newlines(self):
        assert parse_watchlist_input("AAPL\nMSFT\n") == ["AAPL", "MSFT"]

    def test_parse_spaces(self):
        assert parse_watchlist_input("AAPL MSFT GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_mixed(self):
        assert parse_watchlist_input("AAPL, MSFT\nGOOGL TSLA") == [
            "AAPL", "MSFT", "GOOGL", "TSLA"
        ]

    def test_parse_dedup(self):
        assert parse_watchlist_input("AAPL, AAPL, MSFT") == ["AAPL", "MSFT"]

    def test_parse_empty(self):
        assert parse_watchlist_input("") == []


class TestScreenerPick:
    def test_defaults(self):
        pick = ScreenerPick(ticker="AAPL")
        assert pick.signal == "hold"
        assert pick.confidence == 0.0
        assert pick.rationale == ""
        assert pick.position_size_pct == 0.0
        assert pick.time_horizon == ""
        assert pick.key_risks == []
        assert pick.rank == 0


class TestScreenerResult:
    def test_defaults(self):
        result = ScreenerResult()
        assert result.picks == []
        assert result.run_id != ""
        assert result.watchlist == []

    def test_top_n(self):
        picks = [
            ScreenerPick(ticker="AAPL", confidence=80, rank=1),
            ScreenerPick(ticker="MSFT", confidence=70, rank=2),
            ScreenerPick(ticker="GOOGL", confidence=60, rank=3),
        ]
        result = ScreenerResult(picks=picks, watchlist=["AAPL", "MSFT", "GOOGL"])
        top2 = result.top_n(2)
        assert len(top2) == 2
        assert top2[0].ticker == "AAPL"


def _make_config():
    return AppConfig(
        llm=LLMConfig(provider="lmstudio", model="local-model"),
        trading=TradingConfig(
            risk_tolerance="moderate", max_debate_rounds=1, analysis_period_days=90
        ),
    )


def _make_price_data():
    return pd.DataFrame(
        {
            "Open": [1, 2, 3] * 20,
            "High": [2, 3, 4] * 20,
            "Low": [0.5, 1, 2] * 20,
            "Close": [1.5, 2.5, 3.5] * 20,
            "Volume": [100, 200, 300] * 20,
        },
        index=pd.date_range("2025-01-01", periods=60),
    )


class TestOpenTradeScreener:
    @patch("opentrade_ai.screener.TechnicalAnalyzer")
    @patch("opentrade_ai.screener.MarketDataProvider")
    def test_gather_ticker_data_success(self, mock_data_cls, mock_ta_cls):
        config = _make_config()

        mock_data = MagicMock()
        mock_data.get_stock_info.return_value = {"name": "Apple", "sector": "Tech"}
        mock_data.get_recent_news.return_value = []
        mock_data.get_historical_data.return_value = _make_price_data()
        mock_data_cls.return_value = mock_data

        mock_ta = MagicMock()
        mock_ta.compute_indicators.return_value = {"rsi": 55, "price_change_pct": 1.2}
        mock_ta.get_signal_summary.return_value = {"overall": "bullish"}
        mock_ta_cls.return_value = mock_ta

        llm = MagicMock()
        screener = OpenTradeScreener(config=config, llm=llm)
        data = screener._gather_ticker_data("AAPL", "2025-02-15")
        assert data["ticker"] == "AAPL"
        assert data["stock_info"]["name"] == "Apple"

    @patch("opentrade_ai.screener.TechnicalAnalyzer")
    @patch("opentrade_ai.screener.MarketDataProvider")
    def test_gather_ticker_data_failure(self, mock_data_cls, mock_ta_cls):
        config = _make_config()
        mock_data = MagicMock()
        mock_data.get_stock_info.side_effect = ValueError("Invalid ticker")
        mock_data_cls.return_value = mock_data
        mock_ta_cls.return_value = MagicMock()

        llm = MagicMock()
        screener = OpenTradeScreener(config=config, llm=llm)
        data = screener._gather_ticker_data("INVALID", "2025-02-15")
        assert data is None

    @patch("opentrade_ai.screener.TechnicalAnalyzer")
    @patch("opentrade_ai.screener.MarketDataProvider")
    def test_screen_ranks_picks(self, mock_data_cls, mock_ta_cls):
        config = _make_config()

        mock_data = MagicMock()
        mock_data.get_stock_info.return_value = {"name": "Test", "sector": "Tech"}
        mock_data.get_recent_news.return_value = []
        mock_data.get_historical_data.return_value = _make_price_data()
        mock_data_cls.return_value = mock_data

        mock_ta = MagicMock()
        mock_ta.compute_indicators.return_value = {"rsi": 55, "price_change_pct": 1.2}
        mock_ta.get_signal_summary.return_value = {"overall": "bullish"}
        mock_ta_cls.return_value = mock_ta

        llm = MagicMock()
        llm.generate.return_value = (
            '{"picks": ['
            '{"ticker": "AAPL", "signal": "BUY", "confidence": 80, "rationale": "Strong", '
            '"position_size_pct": 3, "time_horizon": "swing", "key_risks": ["macro"]},'
            '{"ticker": "MSFT", "signal": "HOLD", "confidence": 55, "rationale": "OK", '
            '"position_size_pct": 1, "time_horizon": "long", "key_risks": ["competition"]}'
            ']}'
        )

        screener = OpenTradeScreener(config=config, llm=llm)
        result = screener.screen(["AAPL", "MSFT"], date="2025-02-15", top_n=5)

        assert len(result.picks) >= 1
        assert result.picks[0].confidence >= result.picks[-1].confidence

    @patch("opentrade_ai.screener.TechnicalAnalyzer")
    @patch("opentrade_ai.screener.MarketDataProvider")
    def test_screen_freetext_fallback(self, mock_data_cls, mock_ta_cls):
        config = _make_config()

        mock_data = MagicMock()
        mock_data.get_stock_info.return_value = {"name": "Test", "sector": "Tech"}
        mock_data.get_recent_news.return_value = []
        mock_data.get_historical_data.return_value = _make_price_data()
        mock_data_cls.return_value = mock_data

        mock_ta = MagicMock()
        mock_ta.compute_indicators.return_value = {"rsi": 55}
        mock_ta.get_signal_summary.return_value = {"overall": "bullish"}
        mock_ta_cls.return_value = mock_ta

        llm = MagicMock()
        llm.generate.return_value = "AAPL is a BUY with 80% confidence. Strong growth ahead."

        screener = OpenTradeScreener(config=config, llm=llm)
        result = screener.screen(["AAPL"], date="2025-02-15", top_n=5)
        assert isinstance(result, ScreenerResult)
