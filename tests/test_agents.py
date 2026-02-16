from opentrade_ai.agents.base import AgentRole, AnalysisResult
from opentrade_ai.agents.bear_researcher import BearResearcher
from opentrade_ai.agents.bull_researcher import BullResearcher
from opentrade_ai.agents.fundamental_analyst import FundamentalAnalyst
from opentrade_ai.agents.news_analyst import NewsAnalyst
from opentrade_ai.agents.risk_manager import RiskManager
from opentrade_ai.agents.sentiment_analyst import SentimentAnalyst
from opentrade_ai.agents.technical_analyst import TechnicalAnalyst
from opentrade_ai.agents.trader import TraderAgent


class TestAnalysisResult:
    def test_default_values(self):
        result = AnalysisResult(
            agent_role=AgentRole.FUNDAMENTAL_ANALYST,
            ticker="AAPL",
            summary="Test summary",
        )
        assert result.signal == "neutral"
        assert result.confidence == 0.0
        assert result.details == {}

    def test_custom_values(self):
        result = AnalysisResult(
            agent_role=AgentRole.TRADER,
            ticker="MSFT",
            summary="Strong buy",
            signal="buy",
            confidence=85.0,
            details={"key": "value"},
        )
        assert result.signal == "buy"
        assert result.confidence == 85.0
        assert result.details["key"] == "value"


class TestFundamentalAnalyst:
    def test_role(self, mock_llm):
        agent = FundamentalAnalyst(mock_llm)
        assert agent.role == AgentRole.FUNDAMENTAL_ANALYST

    def test_analyze_returns_result(self, mock_llm, sample_stock_info):
        agent = FundamentalAnalyst(mock_llm)
        context = {"stock_info": sample_stock_info}
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.agent_role == AgentRole.FUNDAMENTAL_ANALYST
        mock_llm.generate.assert_called_once()

    def test_analyze_parses_buy_signal(self, mock_llm, sample_stock_info):
        mock_llm.generate.return_value = "Signal: BUY. The stock has strong growth."
        agent = FundamentalAnalyst(mock_llm)
        result = agent.analyze("AAPL", {"stock_info": sample_stock_info})
        assert result.signal == "buy"

    def test_analyze_parses_json_output(self, mock_llm, sample_stock_info):
        mock_llm.generate.return_value = (
            '{"signal": "BUY", "confidence": 77, "summary": "Bullish on valuation."}'
        )
        agent = FundamentalAnalyst(mock_llm)
        result = agent.analyze("AAPL", {"stock_info": sample_stock_info})
        assert result.signal == "buy"
        assert result.confidence == 77
        assert "Bullish" in result.summary

    def test_analyze_with_empty_context(self, mock_llm):
        agent = FundamentalAnalyst(mock_llm)
        result = agent.analyze("AAPL", {})
        assert isinstance(result, AnalysisResult)


class TestSentimentAnalyst:
    def test_role(self, mock_llm):
        agent = SentimentAnalyst(mock_llm)
        assert agent.role == AgentRole.SENTIMENT_ANALYST

    def test_analyze_returns_result(self, mock_llm, sample_news):
        agent = SentimentAnalyst(mock_llm)
        context = {"stock_info": {"name": "Apple"}, "news": sample_news}
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.details["news_count"] == 2

    def test_analyze_with_no_news(self, mock_llm):
        agent = SentimentAnalyst(mock_llm)
        result = agent.analyze("AAPL", {"news": []})
        assert result.details["news_count"] == 0


class TestNewsAnalyst:
    def test_role(self, mock_llm):
        agent = NewsAnalyst(mock_llm)
        assert agent.role == AgentRole.NEWS_ANALYST

    def test_analyze_returns_result(self, mock_llm, sample_news):
        agent = NewsAnalyst(mock_llm)
        context = {"stock_info": {"name": "Apple"}, "news": sample_news}
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.details["news_analyzed"] == 2


class TestTechnicalAnalyst:
    def test_role(self, mock_llm):
        agent = TechnicalAnalyst(mock_llm)
        assert agent.role == AgentRole.TECHNICAL_ANALYST

    def test_analyze_returns_result(self, mock_llm):
        agent = TechnicalAnalyst(mock_llm)
        context = {
            "indicators": {"rsi": 55, "macd": 0.5, "current_price": 150.0},
            "signals": {"rsi": "neutral"},
        }
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.details["indicators"]["rsi"] == 55


class TestBullResearcher:
    def test_role(self, mock_llm):
        agent = BullResearcher(mock_llm)
        assert agent.role == AgentRole.BULL_RESEARCHER

    def test_analyze_returns_result(self, mock_llm):
        agent = BullResearcher(mock_llm)
        context = {"analyst_reports": [], "stock_info": {}}
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.details["perspective"] == "bullish"

    def test_debate(self, mock_llm):
        mock_llm.generate.return_value = "Counter argument: The company has strong moats."
        agent = BullResearcher(mock_llm)
        context = {"analyst_reports": []}
        response = agent.debate("AAPL", context, "The stock is overvalued.")
        assert "Counter argument" in response


class TestBearResearcher:
    def test_role(self, mock_llm):
        agent = BearResearcher(mock_llm)
        assert agent.role == AgentRole.BEAR_RESEARCHER

    def test_analyze_returns_result(self, mock_llm):
        agent = BearResearcher(mock_llm)
        context = {"analyst_reports": [], "stock_info": {}}
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)
        assert result.details["perspective"] == "bearish"

    def test_debate(self, mock_llm):
        mock_llm.generate.return_value = "Counter argument: Valuation is stretched."
        agent = BearResearcher(mock_llm)
        context = {"analyst_reports": []}
        response = agent.debate("AAPL", context, "The stock has strong growth.")
        assert "Counter argument" in response


class TestTraderAgent:
    def test_role(self, mock_llm):
        agent = TraderAgent(mock_llm)
        assert agent.role == AgentRole.TRADER

    def test_analyze_returns_result(self, mock_llm):
        agent = TraderAgent(mock_llm)
        context = {
            "analyst_reports": [],
            "debate_history": [],
            "risk_tolerance": "moderate",
        }
        result = agent.analyze("AAPL", context)
        assert isinstance(result, AnalysisResult)

    def test_analyze_with_reports(self, mock_llm):
        reports = [
            AnalysisResult(
                agent_role=AgentRole.FUNDAMENTAL_ANALYST,
                ticker="AAPL",
                summary="Strong fundamentals",
                signal="buy",
                confidence=75.0,
            )
        ]
        agent = TraderAgent(mock_llm)
        context = {
            "analyst_reports": reports,
            "debate_history": [{"bull": "Growth", "bear": "Valuation"}],
            "risk_tolerance": "moderate",
        }
        result = agent.analyze("AAPL", context)
        assert result.details["analyst_count"] == 1
        assert result.details["debate_rounds"] == 1


class TestRiskManager:
    def test_role(self, mock_llm):
        agent = RiskManager(mock_llm)
        assert agent.role == AgentRole.RISK_MANAGER

    def test_analyze_approve(self, mock_llm):
        mock_llm.generate.return_value = "APPROVE. Risk is within acceptable levels."
        agent = RiskManager(mock_llm)
        trader_decision = AnalysisResult(
            agent_role=AgentRole.TRADER,
            ticker="AAPL",
            summary="Buy recommendation",
            signal="buy",
            confidence=70.0,
        )
        context = {
            "trader_decision": trader_decision,
            "indicators": {"atr": 2.5, "rsi": 55},
            "stock_info": {"beta": 1.2},
            "risk_tolerance": "moderate",
        }
        result = agent.analyze("AAPL", context)
        assert result.signal == "approve"

    def test_analyze_reject(self, mock_llm):
        mock_llm.generate.return_value = "REJECT. Too risky given current volatility."
        agent = RiskManager(mock_llm)
        context = {
            "trader_decision": AnalysisResult(
                agent_role=AgentRole.TRADER,
                ticker="AAPL",
                summary="Buy",
                signal="buy",
                confidence=70.0,
            ),
            "indicators": {},
            "stock_info": {},
            "risk_tolerance": "conservative",
        }
        result = agent.analyze("AAPL", context)
        assert result.signal == "reject"

    def test_analyze_modify(self, mock_llm):
        mock_llm.generate.return_value = "MODIFY. Reduce position size to 2%."
        agent = RiskManager(mock_llm)
        context = {
            "trader_decision": AnalysisResult(
                agent_role=AgentRole.TRADER,
                ticker="AAPL",
                summary="Buy",
                signal="buy",
                confidence=70.0,
            ),
            "indicators": {},
            "stock_info": {},
            "risk_tolerance": "moderate",
        }
        result = agent.analyze("AAPL", context)
        assert result.signal == "modify"
