from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class NewsAnalyst(BaseAgent):
    role = AgentRole.NEWS_ANALYST
    system_prompt = (
        "You are a senior news analyst at a top-tier quantitative trading firm "
        "with deep expertise in information extraction and event-driven trading. "
        "You evaluate how news events translate into stock price impact.\n\n"
        "ANALYSIS FRAMEWORK:\n"
        "1. Materiality Assessment: Rate each news item as HIGH/MEDIUM/LOW "
        "impact. Only HIGH items should significantly affect your signal.\n"
        "2. Catalyst Identification: Earnings surprises, product launches, "
        "M&A activity, regulatory changes, management changes, lawsuits.\n"
        "3. Temporal Impact: Distinguish between short-term noise (1-5 days) "
        "and structural changes (months/quarters).\n"
        "4. Cross-Source Verification: News confirmed by multiple credible "
        "sources (Reuters, Bloomberg, SEC filings) is more reliable.\n"
        "5. SEC Filing Analysis: Recent 10-K/10-Q/8-K filings may contain "
        "material disclosures not yet priced in by the market.\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"signal": "STRONG BUY", "confidence": 82, "summary": '
        '"Three HIGH-impact catalysts identified for NVDA: (1) Reuters reports '
        "Q4 data center revenue up 40% YoY, beating consensus by 12% - this "
        "is structural given AI infrastructure demand. (2) SEC 10-Q filing "
        "shows gross margins expanded to 76%, highest in 5 years. (3) Google "
        "News confirms new partnership with major cloud provider. One negative: "
        "Bloomberg reports potential export restriction tightening, but this "
        "is speculative (MEDIUM impact). Net news flow is strongly positive "
        'with structural tailwinds."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
        '"confidence": <0-100>, "summary": "<your detailed news analysis>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {
            "news_analyzed": len(context.get("news", [])),
            "google_news_count": len(context.get("google_news", [])),
            "sec_filings_count": len(context.get("sec_filings", [])),
        }
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        info = context.get("stock_info", {})
        news = context.get("news", [])
        google_news = context.get("google_news", [])
        sec_filings = context.get("sec_filings", [])

        parts = [
            f"Analyze all news and filings for {ticker} "
            f"({info.get('name', ticker)}).",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
        ]

        parts.append("\n--- YAHOO FINANCE NEWS ---")
        if news:
            for item in news[:8]:
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                summary = item.get("summary", "")
                parts.append(f"- [{publisher}] {title}")
                if summary:
                    parts.append(f"  Summary: {summary[:200]}")
        else:
            parts.append("- No Yahoo Finance news available")

        if google_news:
            parts.append("\n--- GOOGLE NEWS (broader web coverage) ---")
            for item in google_news[:8]:
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                desc = item.get("description", "")
                parts.append(f"- [{publisher}] {title}")
                if desc:
                    parts.append(f"  Description: {desc[:200]}")

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
            "Rate each news item by materiality (HIGH/MEDIUM/LOW). "
            "Identify catalysts, assess temporal impact, and cross-verify "
            "across sources. If SEC filings are provided, check for "
            "material disclosures.\n"
            "Respond with JSON: "
            '{"signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
            '"confidence": <0-100>, "summary": "<your detailed analysis>"}'
        )
        return "\n".join(parts)
