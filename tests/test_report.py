import json
import os
import tempfile

from opentrade_ai.agents.base import AgentRole, AnalysisResult
from opentrade_ai.graph.trading_graph import TradingDecision
from opentrade_ai.report import ReportExporter
from opentrade_ai.screener import ScreenerPick, ScreenerResult


class TestReportExporterSingleTicker:
    def _make_decision(self):
        return TradingDecision(
            ticker="AAPL",
            final_signal="buy",
            confidence=75.0,
            trader_summary="Buy recommendation based on strong fundamentals.",
            risk_assessment="Approved. Low risk.",
            analyst_reports=[
                AnalysisResult(
                    agent_role=AgentRole.FUNDAMENTAL_ANALYST,
                    ticker="AAPL",
                    summary="Strong fundamentals.",
                    signal="buy",
                    confidence=80.0,
                ),
                AnalysisResult(
                    agent_role=AgentRole.TECHNICAL_ANALYST,
                    ticker="AAPL",
                    summary="Bullish trend.",
                    signal="buy",
                    confidence=70.0,
                ),
            ],
            debate_history=[{"bull": "Growth catalysts.", "bear": "Valuation stretched."}],
            stock_info={"name": "Apple Inc.", "sector": "Technology", "current_price": 185.0},
            indicators={"rsi": 55.0, "macd": 0.5},
        )

    def test_to_dict(self):
        decision = self._make_decision()
        exporter = ReportExporter()
        data = exporter.decision_to_dict(decision)
        assert data["ticker"] == "AAPL"
        assert data["final_signal"] == "buy"
        assert data["confidence"] == 75.0
        assert len(data["analyst_reports"]) == 2
        assert data["stock_info"]["name"] == "Apple Inc."

    def test_save_json(self):
        decision = self._make_decision()
        exporter = ReportExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_json(decision, output_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".json")
            with open(path) as f:
                data = json.load(f)
            assert data["ticker"] == "AAPL"

    def test_save_html(self):
        decision = self._make_decision()
        exporter = ReportExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_html(decision, output_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".html")
            with open(path) as f:
                html = f.read()
            assert "AAPL" in html
            assert "buy" in html.lower()


class TestReportExporterScreener:
    def _make_screener_result(self):
        return ScreenerResult(
            picks=[
                ScreenerPick(
                    ticker="AAPL", signal="buy", confidence=80, rank=1,
                    rationale="Strong growth.", position_size_pct=3.0,
                    time_horizon="swing", key_risks=["macro"],
                ),
                ScreenerPick(
                    ticker="MSFT", signal="hold", confidence=55, rank=2,
                    rationale="Fair value.", position_size_pct=1.0,
                    time_horizon="long", key_risks=["competition"],
                ),
            ],
            watchlist=["AAPL", "MSFT"],
        )

    def test_screener_to_dict(self):
        result = self._make_screener_result()
        exporter = ReportExporter()
        data = exporter.screener_to_dict(result)
        assert data["watchlist"] == ["AAPL", "MSFT"]
        assert len(data["picks"]) == 2

    def test_save_screener_json(self):
        result = self._make_screener_result()
        exporter = ReportExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_screener_json(result, output_dir=tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["picks"]) == 2

    def test_save_screener_html(self):
        result = self._make_screener_result()
        exporter = ReportExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_screener_html(result, output_dir=tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                html = f.read()
            assert "AAPL" in html
            assert "MSFT" in html
