from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class BullResearcher(BaseAgent):
    role = AgentRole.BULL_RESEARCHER
    system_prompt = (
        "You are the lead bullish researcher at a top-tier investment research firm. "
        "Your job is to construct the most compelling BULL CASE possible, as if you "
        "are presenting to a portfolio manager who controls $1B in capital.\n\n"
        "BULL CASE FRAMEWORK:\n"
        "1. Growth Catalysts: New products, TAM expansion, market share gains, "
        "secular tailwinds (AI, cloud, EVs, etc.).\n"
        "2. Valuation Opportunity: Is the stock cheap relative to growth? "
        "Compare forward P/E to expected EPS growth (PEG ratio).\n"
        "3. Competitive Moat: Network effects, switching costs, brand, scale "
        "advantages, patents, regulatory barriers.\n"
        "4. Financial Strength: Strong balance sheet, rising FCF, margin "
        "expansion trajectory, capital return (buybacks/dividends).\n"
        "5. Technical Setup: Is the chart supportive? Breakout patterns, "
        "accumulation signals, momentum confirmation.\n"
        "6. Counter-Bear Arguments: Directly rebut the strongest bearish "
        "concerns with data and logic.\n\n"
        "Be persuasive, specific, and data-driven. Cite exact numbers from "
        "the analyst reports when available. Your goal is to find alpha that "
        "the bears are missing."
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {"perspective": "bullish"}
        return result

    def debate(self, ticker: str, context: dict, bear_argument: str) -> str:
        prompt = (
            f"The bear researcher argues against {ticker}:\n\n"
            f"{bear_argument}\n\n"
            "Counter these bearish arguments with your strongest bull case. "
            "Use data from the analyst reports to support your position."
        )
        analyst_summary = self._format_analyst_reports(context)
        if analyst_summary:
            prompt += f"\n\nAnalyst Reports:\n{analyst_summary}"
        return self.llm.generate(prompt, self.system_prompt)

    def _build_prompt(self, ticker: str, context: dict) -> str:
        parts = [
            f"Build the strongest BULL CASE for {ticker}.",
            "\nAnalyst reports available:",
        ]
        parts.append(self._format_analyst_reports(context))
        parts.append(
            "\nPresent your bullish thesis. What are the key reasons to BUY this stock? "
            "Include growth catalysts, valuation arguments, and positive trends."
        )
        return "\n".join(parts)

    def _format_analyst_reports(self, context: dict) -> str:
        reports = context.get("analyst_reports", [])
        if not reports:
            return "No analyst reports available."
        parts = []
        for report in reports:
            role_val = report.agent_role
            role = role_val.value if hasattr(role_val, "value") else role_val
            parts.append(f"\n[{role}] Signal: {report.signal}")
            parts.append(f"Summary: {report.summary[:500]}")
        return "\n".join(parts)
