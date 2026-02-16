from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class FundamentalAnalyst(BaseAgent):
    role = AgentRole.FUNDAMENTAL_ANALYST
    system_prompt = (
        "You are a senior fundamental analyst at a top-tier quantitative trading firm "
        "with 15+ years of experience in equity research. Your analysis must be "
        "rigorous, data-driven, and actionable.\n\n"
        "ANALYSIS FRAMEWORK:\n"
        "1. Valuation: Compare P/E, Forward P/E to sector norms. "
        "Flag if >20% above/below peers.\n"
        "2. Profitability: Assess profit margins, ROE, and free cash flow yield. "
        "Deteriorating margins are a red flag.\n"
        "3. Growth: Evaluate revenue growth trajectory. "
        "Deceleration from >20% to <10% is material.\n"
        "4. Balance Sheet: Debt/Equity >1.5 in non-financial sectors is concerning.\n"
        "5. Competitive Position: Moat durability, market share trends.\n"
        "6. SEC Filings: If recent 10-K/10-Q data is provided, incorporate key "
        "findings (revenue trends, risk factors, management discussion).\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"signal": "BUY", "confidence": 78, "summary": '
        '"AAPL trades at 28x trailing earnings vs. sector median of 32x, '
        "suggesting slight undervaluation. Revenue growth of 8% YoY is "
        "modest but supported by 38% profit margins (best-in-class) and "
        "$100B+ FCF. Debt/Equity of 1.8 is elevated but manageable given "
        "cash reserves. Services segment growing 15% provides recurring "
        "revenue diversification. Key risk: iPhone revenue concentration "
        '(52% of sales)."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
        '"confidence": <0-100>, "summary": "<your detailed analysis>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {
            "stock_info": context.get("stock_info", {}),
            "financials_available": "financials" in context,
            "sec_filings_count": len(context.get("sec_filings", [])),
        }
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        info = context.get("stock_info", {})
        parts = [
            f"Perform a comprehensive fundamental analysis of {ticker} "
            f"({info.get('name', ticker)}).",
            f"\nSector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            "\n--- VALUATION & FINANCIAL METRICS ---",
        ]

        if info.get("market_cap"):
            parts.append(f"Market Cap: ${info['market_cap']:,.0f}")
        if info.get("pe_ratio"):
            parts.append(f"P/E Ratio (Trailing): {info['pe_ratio']:.2f}")
        if info.get("forward_pe"):
            parts.append(f"Forward P/E: {info['forward_pe']:.2f}")
        if info.get("revenue_growth"):
            parts.append(f"Revenue Growth (YoY): {info['revenue_growth']:.2%}")
        if info.get("profit_margins"):
            parts.append(f"Profit Margins: {info['profit_margins']:.2%}")
        if info.get("debt_to_equity"):
            parts.append(f"Debt/Equity: {info['debt_to_equity']:.2f}")
        if info.get("return_on_equity"):
            parts.append(f"ROE: {info['return_on_equity']:.2%}")
        if info.get("free_cash_flow"):
            parts.append(f"Free Cash Flow: ${info['free_cash_flow']:,.0f}")
        if info.get("dividend_yield"):
            parts.append(f"Dividend Yield: {info['dividend_yield']:.2%}")
        if info.get("beta"):
            parts.append(f"Beta: {info['beta']:.2f}")
        if info.get("current_price"):
            parts.append(f"Current Price: ${info['current_price']:.2f}")
        w52h = info.get("52_week_high")
        w52l = info.get("52_week_low")
        if w52h and w52l:
            parts.append(f"52-Week Range: ${w52l:.2f} - ${w52h:.2f}")

        sec_filings = context.get("sec_filings", [])
        if sec_filings:
            parts.append("\n--- RECENT SEC FILINGS ---")
            for filing in sec_filings[:5]:
                parts.append(
                    f"- {filing.get('form', 'N/A')} filed "
                    f"{filing.get('filing_date', 'N/A')}: "
                    f"{filing.get('description', 'N/A')}"
                )

        parts.append(
            "\n--- INSTRUCTIONS ---\n"
            "Evaluate across all five dimensions: valuation, profitability, "
            "growth, balance sheet, competitive position. Cite specific "
            "numbers. If SEC filings are provided, incorporate findings.\n"
            "Respond with JSON: "
            '{"signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
            '"confidence": <0-100>, "summary": "<your detailed analysis>"}'
        )
        return "\n".join(parts)
