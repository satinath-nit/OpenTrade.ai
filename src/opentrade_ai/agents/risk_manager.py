from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class RiskManager(BaseAgent):
    role = AgentRole.RISK_MANAGER
    system_prompt = (
        "You are the Chief Risk Officer at a top-tier quantitative trading firm "
        "responsible for protecting $500M in capital. You have final veto power "
        "over all trades. Your job is to protect against catastrophic losses.\n\n"
        "RISK ASSESSMENT FRAMEWORK:\n"
        "1. Volatility Risk: ATR >3% of price = HIGH volatility. RSI extremes "
        "(>80 or <20) indicate potential mean-reversion risk.\n"
        "2. Position Sizing: Max 5% of portfolio for any single position. "
        "For HIGH volatility stocks, reduce to 2-3%. Beta >1.5 = extra caution.\n"
        "3. Liquidity Risk: Low-volume stocks risk slippage. Ensure average "
        "daily volume > 10x planned position size.\n"
        "4. Correlation Risk: Avoid concentrating in correlated positions. "
        "Check sector exposure.\n"
        "5. Downside Scenario: What is the max drawdown if thesis is wrong? "
        "Stop-loss must limit loss to <2% of total portfolio.\n"
        "6. Debt/Leverage: Debt-to-equity >2.0 is a yellow flag; >4.0 is red.\n\n"
        "DECISIONS:\n"
        "- APPROVE: Risk is acceptable, proceed as proposed.\n"
        "- MODIFY: Adjust position size, add stop-loss, or tighten conditions.\n"
        "- REJECT: Risk is too high; do not trade.\n\n"
        "EXAMPLE RESPONSE:\n"
        "Decision: MODIFY\n"
        "Risk Level: MEDIUM\n"
        "Reasoning: Trader proposes 5% position in TSLA (BUY, 72% confidence). "
        "However, ATR is 4.2% of price (HIGH volatility) and beta is 1.8. "
        "Reducing position to 2.5% and requiring stop-loss at $220 (2x ATR "
        "below entry). Debt-to-equity at 0.9 is acceptable. RSI at 65 is "
        "not overbought but approaching caution zone.\n"
        "Key Risks: (1) High volatility amplifies drawdown, (2) Beta >1.5 "
        "means market downturn would hit this position harder.\n"
        "Adjusted Stop-Loss: $220 (limits portfolio loss to 1.1%)."
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_risk_response(ticker, response)
        trader_dec = context.get("trader_decision")
        trader_signal = trader_dec.signal if trader_dec else ""
        result.details = {"trader_signal": trader_signal}
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        trader_decision = context.get("trader_decision")
        indicators = context.get("indicators", {})

        parts = [f"Review the trading proposal for {ticker}.", "\n--- TRADER PROPOSAL ---"]

        if trader_decision:
            parts.append(f"Signal: {trader_decision.signal}")
            parts.append(f"Confidence: {trader_decision.confidence}%")
            parts.append(f"Summary: {trader_decision.summary[:500]}")

        parts.append("\n--- RISK METRICS ---")
        if indicators.get("atr") is not None:
            parts.append(f"ATR (Volatility): {indicators['atr']:.4f}")
        if indicators.get("rsi") is not None:
            parts.append(f"RSI: {indicators['rsi']:.2f}")
        if indicators.get("volume_trend") is not None:
            parts.append(f"Volume Trend: {indicators['volume_trend']:.2f}x")

        stock_info = context.get("stock_info", {})
        if stock_info.get("beta"):
            parts.append(f"Beta: {stock_info['beta']:.2f}")
        if stock_info.get("debt_to_equity"):
            parts.append(f"Debt/Equity: {stock_info['debt_to_equity']:.2f}")

        risk_tolerance = context.get("risk_tolerance", "moderate")
        parts.append(f"\nPortfolio Risk Tolerance: {risk_tolerance}")
        parts.append(
            "\nProvide your risk assessment:\n"
            "1. Decision: APPROVE / MODIFY / REJECT\n"
            "2. Risk level: LOW / MEDIUM / HIGH\n"
            "3. If MODIFY: suggested adjustments\n"
            "4. Key risk factors\n"
            "5. Recommended stop-loss level if applicable"
        )
        return "\n".join(parts)

    def _parse_risk_response(self, ticker: str, response: str) -> AnalysisResult:
        lower = response.lower()
        if "reject" in lower:
            signal = "reject"
            confidence = 80.0
        elif "modify" in lower:
            signal = "modify"
            confidence = 65.0
        elif "approve" in lower:
            signal = "approve"
            confidence = 75.0
        else:
            signal = "review"
            confidence = 50.0

        return AnalysisResult(
            agent_role=self.role,
            ticker=ticker,
            summary=response,
            signal=signal,
            confidence=confidence,
        )
