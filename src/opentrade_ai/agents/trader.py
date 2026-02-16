from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class TraderAgent(BaseAgent):
    role = AgentRole.TRADER
    system_prompt = (
        "You are a senior portfolio trader at a top-tier quantitative trading firm "
        "managing a $500M equity book. You receive analyst reports, bull/bear debate "
        "transcripts, and must synthesize everything into a final trading decision.\n\n"
        "DECISION FRAMEWORK:\n"
        "1. Signal Consensus: Weight analyst signals by confidence. If 3/4 analysts "
        "agree, that's a strong consensus. Disagreement requires careful judgment.\n"
        "2. Bull/Bear Balance: Which side presented stronger evidence in the debate? "
        "Did either side fail to address key counterarguments?\n"
        "3. Risk/Reward Ratio: Target at least 2:1 reward-to-risk for BUY signals. "
        "Calculate implied upside vs. downside from support/resistance levels.\n"
        "4. Position Sizing: Conservative=1-2%, Moderate=2-5%, Aggressive=5-10% "
        "of portfolio. Scale with confidence level.\n"
        "5. Time Horizon: Specify if this is a swing trade (1-4 weeks), "
        "position trade (1-6 months), or long-term hold (6+ months).\n"
        "6. Entry/Exit Strategy: Suggest entry zone, stop-loss, and profit target.\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"signal": "BUY", "confidence": 76, "summary": '
        '"Synthesizing 4 analyst reports and 2 debate rounds for AAPL: '
        "3/4 analysts bullish (fundamental, technical, sentiment) with 1 "
        "neutral (news). Bull researcher's argument on Services growth was "
        "more compelling than Bear's margin compression concern (margins "
        "actually expanded last quarter). Risk/reward is 2.5:1 with entry "
        "at $184, stop at $178 (SMA50), target $199. Recommended position: "
        "3% of portfolio (moderate risk tolerance). Time horizon: 2-3 months "
        '(position trade through next earnings)."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
        '"confidence": <0-100>, "summary": "<decision with position size, '
        'time horizon, entry/exit, and top 3 factors>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {
            "analyst_count": len(context.get("analyst_reports", [])),
            "debate_rounds": len(context.get("debate_history", [])),
        }
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        parts = [f"Make a trading decision for {ticker}.", "\n--- ANALYST REPORTS ---"]

        for report in context.get("analyst_reports", []):
            role_val = report.agent_role
            role = role_val.value if hasattr(role_val, "value") else role_val
            parts.append(f"\n[{role}] Signal: {report.signal} | Confidence: {report.confidence}%")
            parts.append(f"Summary: {report.summary[:400]}")

        debate_history = context.get("debate_history", [])
        if debate_history:
            parts.append("\n--- RESEARCH DEBATE ---")
            for i, entry in enumerate(debate_history):
                parts.append(f"\nRound {i + 1}:")
                parts.append(f"Bull: {entry.get('bull', '')[:300]}")
                parts.append(f"Bear: {entry.get('bear', '')[:300]}")

        risk_tolerance = context.get("risk_tolerance", "moderate")
        parts.append(f"\nRisk Tolerance: {risk_tolerance}")
        parts.append(
            "\nBased on all the above, respond with JSON: "
            '{"signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
            '"confidence": <0-100>, "summary": "<decision with position size, '
            'time horizon, and top 3 factors>"}'
        )
        return "\n".join(parts)
