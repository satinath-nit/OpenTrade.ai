from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class SentimentAnalyst(BaseAgent):
    role = AgentRole.SENTIMENT_ANALYST
    system_prompt = (
        "You are a senior sentiment analyst at a top-tier quantitative trading firm "
        "specializing in alternative data and behavioral finance. You synthesize "
        "signals from news tone, search interest trends, and market narrative.\n\n"
        "ANALYSIS FRAMEWORK:\n"
        "1. News Sentiment: Classify headlines as positive/negative/neutral. "
        "Weight by source credibility (Reuters/Bloomberg > blogs).\n"
        "2. Narrative Momentum: Is the stock in a positive/negative news cycle? "
        "How many days has the current narrative persisted?\n"
        "3. Search Interest (Google Trends): Rising search interest can precede "
        "price moves. Spikes >50% above average may indicate retail attention.\n"
        "4. Contrarian Signals: Extreme bullish sentiment can be a sell signal "
        "(crowded trade). Extreme bearish sentiment can be a buy signal.\n"
        "5. Catalyst Assessment: Are upcoming events (earnings, FDA, etc.) "
        "driving sentiment? Distinguish hype from substance.\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"signal": "BUY", "confidence": 72, "summary": '
        '"Sentiment is cautiously bullish for MSFT. 7 of 10 recent headlines '
        "are positive, led by strong Azure cloud growth coverage from "
        "Reuters and Bloomberg. Google Trends shows 'MSFT stock' search "
        "interest rising 30% over 3 months, suggesting growing retail "
        "interest without euphoria. No extreme crowding signals detected. "
        "Upcoming earnings in 2 weeks could be a catalyst given positive "
        'pre-earnings drift pattern."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
        '"confidence": <0-100>, "summary": "<your detailed sentiment analysis>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {
            "news_count": len(context.get("news", [])),
            "google_news_count": len(context.get("google_news", [])),
            "google_trends_trend": context.get("google_trends", {}).get(
                "trend", "disabled"
            ),
        }
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        info = context.get("stock_info", {})
        news = context.get("news", [])
        google_news = context.get("google_news", [])
        google_trends = context.get("google_trends", {})

        parts = [
            f"Perform a comprehensive sentiment analysis for {ticker} "
            f"({info.get('name', ticker)}).",
            f"Sector: {info.get('sector', 'N/A')}",
        ]

        parts.append("\n--- YAHOO FINANCE NEWS ---")
        if news:
            for item in news[:8]:
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                parts.append(f"- [{publisher}] {title}")
        else:
            parts.append("- No Yahoo Finance news available")

        if google_news:
            parts.append("\n--- GOOGLE NEWS (broader web) ---")
            for item in google_news[:8]:
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                parts.append(f"- [{publisher}] {title}")

        if google_trends and google_trends.get("trend") not in (
            "disabled", "error", "no_data"
        ):
            parts.append("\n--- GOOGLE TRENDS (search interest) ---")
            parts.append(f"Keyword: {google_trends.get('keyword', 'N/A')}")
            parts.append(
                f"Average Interest: {google_trends.get('average_interest', 0)}/100"
            )
            parts.append(
                f"Current Interest: {google_trends.get('current_interest', 0)}/100"
            )
            parts.append(f"Trend Direction: {google_trends.get('trend', 'N/A')}")

        parts.append(
            "\n--- INSTRUCTIONS ---\n"
            "Synthesize all sentiment signals above. Classify the overall "
            "sentiment, assess narrative momentum, check for contrarian "
            "signals, and identify upcoming catalysts.\n"
            "Respond with JSON: "
            '{"signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
            '"confidence": <0-100>, "summary": "<your detailed analysis>"}'
        )
        return "\n".join(parts)
