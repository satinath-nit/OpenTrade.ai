from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class BearResearcher(BaseAgent):
    role = AgentRole.BEAR_RESEARCHER
    system_prompt = (
        "You are the lead bearish researcher at a top-tier investment research firm. "
        "Your job is to construct the most compelling BEAR CASE possible, as if you "
        "are presenting short-sale ideas to a hedge fund manager.\n\n"
        "BEAR CASE FRAMEWORK:\n"
        "1. Overvaluation Risk: Is the stock priced for perfection? Compare P/E, "
        "P/S, EV/EBITDA to historical averages and peers. Flag >30% premium.\n"
        "2. Growth Deceleration: Revenue growth slowing QoQ? Guidance cuts? "
        "Market saturation signals? Customer churn increasing?\n"
        "3. Competitive Threats: New entrants, pricing pressure, technology "
        "disruption, commoditization of the product/service.\n"
        "4. Financial Red Flags: Rising debt, declining margins, negative FCF, "
        "aggressive accounting, insider selling.\n"
        "5. Macro/Regulatory Headwinds: Interest rate sensitivity, regulatory "
        "scrutiny, geopolitical risks, supply chain vulnerabilities.\n"
        "6. Counter-Bull Arguments: Directly challenge the strongest bullish "
        "thesis points with evidence.\n\n"
        "Be rigorous, skeptical, and data-driven. Cite exact numbers from "
        "the analyst reports when available. Your goal is to identify risks "
        "that the bulls are ignoring or underweighting."
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {"perspective": "bearish"}
        return result

    def debate(self, ticker: str, context: dict, bull_argument: str) -> str:
        prompt = (
            f"The bull researcher argues for {ticker}:\n\n"
            f"{bull_argument}\n\n"
            "Counter these bullish arguments with your strongest bear case. "
            "Use data from the analyst reports to support your position."
        )
        analyst_summary = self._format_analyst_reports(context)
        if analyst_summary:
            prompt += f"\n\nAnalyst Reports:\n{analyst_summary}"
        return self.llm.generate(prompt, self.system_prompt)

    def _build_prompt(self, ticker: str, context: dict) -> str:
        parts = [
            f"Build the strongest BEAR CASE against {ticker}.",
            "\nAnalyst reports available:",
        ]
        parts.append(self._format_analyst_reports(context))
        parts.append(
            "\nPresent your bearish thesis. What are the key risks and reasons to SELL? "
            "Include overvaluation concerns, competitive threats, and negative trends."
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
