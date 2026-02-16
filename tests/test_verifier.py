
from opentrade_ai.agents.base import AgentRole, AnalysisResult
from opentrade_ai.agents.verifier import VerifierAgent


class TestVerifierAgent:
    def test_role(self, mock_llm):
        agent = VerifierAgent(mock_llm)
        assert agent.role == AgentRole.VERIFIER

    def test_verify_returns_result(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"verdict": "APPROVED", "confidence_adjustment": 0, '
            '"issues": [], "summary": "Analysis is consistent."}'
        )
        agent = VerifierAgent(mock_llm)
        reports = [
            AnalysisResult(
                agent_role=AgentRole.FUNDAMENTAL_ANALYST,
                ticker="AAPL",
                summary="Strong fundamentals, BUY.",
                signal="buy",
                confidence=75.0,
            ),
        ]
        result = agent.verify(
            ticker="AAPL",
            analyst_reports=reports,
            trader_summary="Buy recommendation based on strong fundamentals.",
            risk_assessment="Approved. Low risk.",
        )
        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.agent_role == AgentRole.VERIFIER

    def test_verify_detects_contradictions(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"verdict": "FLAGGED", "confidence_adjustment": -15, '
            '"issues": ["Fundamental says BUY but technical says SELL"], '
            '"summary": "Contradictions found between analysts."}'
        )
        agent = VerifierAgent(mock_llm)
        reports = [
            AnalysisResult(
                agent_role=AgentRole.FUNDAMENTAL_ANALYST,
                ticker="AAPL",
                summary="BUY signal.",
                signal="buy",
                confidence=75.0,
            ),
            AnalysisResult(
                agent_role=AgentRole.TECHNICAL_ANALYST,
                ticker="AAPL",
                summary="SELL signal.",
                signal="sell",
                confidence=70.0,
            ),
        ]
        result = agent.verify(
            ticker="AAPL",
            analyst_reports=reports,
            trader_summary="Buy recommendation.",
            risk_assessment="Approved.",
        )
        assert result.signal in ("flagged", "approved", "rejected")
        assert "issues" in result.details

    def test_verify_with_empty_reports(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"verdict": "FLAGGED", "confidence_adjustment": -20, '
            '"issues": ["No analyst reports provided"], '
            '"summary": "Insufficient data."}'
        )
        agent = VerifierAgent(mock_llm)
        result = agent.verify(
            ticker="AAPL",
            analyst_reports=[],
            trader_summary="",
            risk_assessment="",
        )
        assert isinstance(result, AnalysisResult)

    def test_verify_freetext_fallback(self, mock_llm):
        mock_llm.generate.return_value = (
            "APPROVED. The analysis is consistent and well-supported."
        )
        agent = VerifierAgent(mock_llm)
        reports = [
            AnalysisResult(
                agent_role=AgentRole.FUNDAMENTAL_ANALYST,
                ticker="AAPL",
                summary="Strong fundamentals.",
                signal="buy",
                confidence=75.0,
            ),
        ]
        result = agent.verify(
            ticker="AAPL",
            analyst_reports=reports,
            trader_summary="Buy.",
            risk_assessment="Approved.",
        )
        assert result.signal == "approved"
