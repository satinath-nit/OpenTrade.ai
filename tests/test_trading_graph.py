from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from opentrade_ai.agents.base import AgentRole
from opentrade_ai.config import AppConfig, LLMConfig, TradingConfig
from opentrade_ai.graph.trading_graph import (
    OpenTradeGraph,
    StepResult,
    TradingDecision,
    _merge_lists,
)


class TestMergeLists:
    def test_merge_empty(self):
        assert _merge_lists([], []) == []

    def test_merge_nonempty(self):
        assert _merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]


class TestStepResult:
    def test_defaults(self):
        step = StepResult(step_name="test")
        assert step.status == "pending"
        assert step.data == {}
        assert step.error == ""


class TestTradingDecision:
    def test_defaults(self):
        decision = TradingDecision(ticker="AAPL")
        assert decision.final_signal == "hold"
        assert decision.confidence == 0.0
        assert decision.analyst_reports == []
        assert decision.debate_history == []

    def test_custom_values(self):
        decision = TradingDecision(
            ticker="AAPL",
            final_signal="buy",
            confidence=75.0,
        )
        assert decision.final_signal == "buy"
        assert decision.confidence == 75.0


class TestOpenTradeGraph:
    @pytest.fixture
    def mock_config(self):
        return AppConfig(
            llm=LLMConfig(provider="ollama", model="llama3"),
            trading=TradingConfig(max_debate_rounds=1, analysis_period_days=90),
        )

    @pytest.fixture
    def mock_price_data(self):
        dates = pd.date_range("2025-01-01", periods=60, freq="B")
        np.random.seed(42)
        prices = 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        return pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
            },
            index=dates,
        )

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_graph_has_compiled_graph(self, mock_data_cls, mock_llm_cls, mock_config):
        graph = OpenTradeGraph(mock_config)
        assert graph.graph is not None

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_emit_step(self, mock_data_cls, mock_llm_cls, mock_config):
        steps_received = []
        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        graph._emit_step("Test Step", "completed", {"key": "val"})
        assert len(steps_received) == 1
        assert steps_received[0].step_name == "Test Step"
        assert steps_received[0].status == "completed"

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_emit_step_callback_failure_does_not_raise(
        self, mock_data_cls, mock_llm_cls, mock_config
    ):
        def bad_callback(_step):
            raise RuntimeError("boom")

        graph = OpenTradeGraph(mock_config, on_step=bad_callback)
        graph._emit_step("Test Step", "completed", {"key": "val"})

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_fetch_data(self, mock_data_cls, mock_llm_cls, mock_config, mock_price_data):
        mock_data = MagicMock()
        mock_data.get_historical_data.return_value = mock_price_data
        mock_data.get_stock_info.return_value = {"name": "Apple", "sector": "Tech"}
        mock_data.get_recent_news.return_value = [{"title": "News"}]
        mock_data_cls.return_value = mock_data

        graph = OpenTradeGraph(mock_config)
        state = {"ticker": "AAPL", "analysis_period_days": 90}
        result = graph._node_fetch_data(state)

        assert "stock_info" in result
        assert "indicators" in result
        assert "price_data" in result
        assert result["stock_info"]["name"] == "Apple"

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_fetch_data_emits_data_sources(
        self, mock_data_cls, mock_llm_cls, mock_config, mock_price_data
    ):
        steps_received = []
        mock_data = MagicMock()
        mock_data.get_historical_data.return_value = mock_price_data
        mock_data.get_stock_info.return_value = {"name": "Apple", "sector": "Tech"}
        mock_data.get_recent_news.return_value = [
            {"title": "Apple rises", "publisher": "Reuters"}
        ]
        mock_data.get_google_news.return_value = [
            {"title": "Apple stock up", "url": "https://example.com/1"}
        ]
        mock_data.get_sec_filings.return_value = [
            {"form": "10-K", "filing_date": "2025-01-01", "url": "https://sec.gov/1"}
        ]
        mock_data.get_google_trends.return_value = {
            "keyword": "Apple stock",
            "trend": "rising",
            "average_interest": 72.0,
            "current_interest": 85.0,
        }
        mock_data_cls.return_value = mock_data

        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        state = {"ticker": "AAPL", "analysis_period_days": 90}
        graph._node_fetch_data(state)

        completed = [s for s in steps_received if s.status == "completed"]
        assert len(completed) == 1
        data = completed[0].data
        assert "data_sources" in data
        sources = {ds["name"]: ds for ds in data["data_sources"]}
        assert "Yahoo Finance" in sources
        assert sources["Yahoo Finance"]["status"] == "ok"
        assert "Google News" in sources
        assert sources["Google News"]["items"] == 1
        assert "SEC EDGAR" in sources
        assert sources["SEC EDGAR"]["items"] == 1
        assert "Google Trends" in sources
        assert sources["Google Trends"]["trend"] == "rising"

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_research_debate_emits_details(
        self, mock_data_cls, mock_llm_cls, mock_config
    ):
        steps_received = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Bullish outlook. Strong fundamentals."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        state = {
            "ticker": "AAPL",
            "analyst_reports": [],
            "stock_info": {},
            "indicators": {},
            "max_debate_rounds": 1,
        }
        graph._node_research_debate(state)

        completed = [s for s in steps_received if s.status == "completed"]
        assert len(completed) == 1
        data = completed[0].data
        assert "rounds" in data
        assert "bull_signal" in data
        assert "bear_signal" in data

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_trader_decision_emits_details(
        self, mock_data_cls, mock_llm_cls, mock_config
    ):
        steps_received = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "BUY signal with 70% confidence."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        state = {
            "ticker": "AAPL",
            "analyst_reports": [],
            "debate_history": [],
            "risk_tolerance": "moderate",
            "stock_info": {},
            "indicators": {},
        }
        graph._node_trader_decision(state)

        completed = [s for s in steps_received if s.status == "completed"]
        assert len(completed) == 1
        data = completed[0].data
        assert "signal" in data
        assert "confidence" in data
        assert "inputs_used" in data

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_risk_review_emits_details(
        self, mock_data_cls, mock_llm_cls, mock_config
    ):
        steps_received = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "APPROVE the trade."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        state = {
            "ticker": "AAPL",
            "trader_summary": "Buy",
            "trader_signal": "buy",
            "trader_confidence": 70.0,
            "final_signal": "buy",
            "final_confidence": 70.0,
            "indicators": {},
            "stock_info": {},
            "risk_tolerance": "moderate",
        }
        graph._node_risk_review(state)

        completed = [s for s in steps_received if s.status == "completed"]
        assert len(completed) == 1
        data = completed[0].data
        assert "risk_signal" in data
        assert "risk_tolerance" in data
        assert "final_signal" in data
        assert "final_confidence" in data

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_verification_emits_details(
        self, mock_data_cls, mock_llm_cls, mock_config
    ):
        steps_received = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "PASS. All analyses are consistent."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config, on_step=lambda s: steps_received.append(s))
        state = {
            "ticker": "AAPL",
            "analyst_reports": [],
            "trader_summary": "Buy AAPL",
            "risk_assessment": "Approved",
            "final_confidence": 70.0,
        }
        graph._node_verification(state)

        completed = [s for s in steps_received if s.status == "completed"]
        assert len(completed) == 1
        data = completed[0].data
        assert "verdict" in data
        assert "issues_count" in data
        assert "confidence_adjustment" in data
        assert "inputs_reviewed" in data

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_run_analysts(self, mock_data_cls, mock_llm_cls, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Signal: BUY. Strong growth outlook."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config)
        state = {
            "ticker": "AAPL",
            "stock_info": {"name": "Apple"},
            "indicators": {},
            "signals": {},
            "news": [],
            "google_news": [],
            "google_trends": {},
            "sec_filings": [],
        }
        result = graph._node_run_analysts(state)

        assert len(result["analyst_reports"]) == 4
        roles = {r.agent_role for r in result["analyst_reports"]}
        assert AgentRole.FUNDAMENTAL_ANALYST in roles
        assert AgentRole.SENTIMENT_ANALYST in roles
        assert AgentRole.NEWS_ANALYST in roles
        assert AgentRole.TECHNICAL_ANALYST in roles

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_run_single_analyst(self, mock_data_cls, mock_llm_cls, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Sentiment is bullish based on recent news."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config)
        result = graph._run_single_analyst(
            "Sentiment Analysis",
            graph.sentiment_analyst,
            "AAPL",
            {"stock_info": {}, "news": []},
        )

        assert result.agent_role == AgentRole.SENTIMENT_ANALYST
        assert result.confidence > 0

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_trader_decision(self, mock_data_cls, mock_llm_cls, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Decision: BUY with 75% confidence."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config)
        state = {
            "ticker": "AAPL",
            "analyst_reports": [],
            "debate_history": [],
            "risk_tolerance": "moderate",
            "stock_info": {},
            "indicators": {},
        }
        result = graph._node_trader_decision(state)

        assert "trader_summary" in result
        assert "final_signal" in result
        assert "final_confidence" in result

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_risk_review_approve(self, mock_data_cls, mock_llm_cls, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "APPROVE the trade. Risk is acceptable."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config)
        state = {
            "ticker": "AAPL",
            "trader_summary": "Buy",
            "trader_signal": "buy",
            "trader_confidence": 70.0,
            "final_signal": "buy",
            "final_confidence": 70.0,
            "indicators": {},
            "stock_info": {},
            "risk_tolerance": "moderate",
        }
        result = graph._node_risk_review(state)

        assert result["final_signal"] == "buy"
        assert result["final_confidence"] == 70.0

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_node_risk_review_reject(self, mock_data_cls, mock_llm_cls, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "REJECT. Too much risk."
        mock_llm_cls.return_value = mock_llm

        graph = OpenTradeGraph(mock_config)
        state = {
            "ticker": "AAPL",
            "trader_summary": "Buy",
            "trader_signal": "buy",
            "trader_confidence": 70.0,
            "final_signal": "buy",
            "final_confidence": 70.0,
            "indicators": {},
            "stock_info": {},
            "risk_tolerance": "conservative",
        }
        result = graph._node_risk_review(state)

        assert result["final_signal"] == "hold"
        assert result["final_confidence"] == max(70.0 * 0.5, 20.0)

    @patch("opentrade_ai.graph.trading_graph.LLMProvider")
    @patch("opentrade_ai.graph.trading_graph.MarketDataProvider")
    def test_propagate_end_to_end(self, mock_data_cls, mock_llm_cls, mock_config, mock_price_data):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "BUY signal with 70% confidence. The stock shows strong growth."
        )
        mock_llm_cls.return_value = mock_llm

        mock_data = MagicMock()
        mock_data.get_historical_data.return_value = mock_price_data
        mock_data.get_stock_info.return_value = {"name": "Apple", "sector": "Technology"}
        mock_data.get_recent_news.return_value = [{"title": "Good news"}]
        mock_data_cls.return_value = mock_data

        graph = OpenTradeGraph(mock_config)
        decision = graph.propagate("AAPL", "2025-02-15")

        assert isinstance(decision, TradingDecision)
        assert decision.ticker == "AAPL"
        assert decision.final_signal in (
            "strong_buy", "buy", "hold", "sell", "strong_sell", "neutral"
        )
        assert len(decision.analyst_reports) > 0
        assert decision.stock_info["name"] == "Apple"
