from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class TechnicalAnalyst(BaseAgent):
    role = AgentRole.TECHNICAL_ANALYST
    system_prompt = (
        "You are a senior technical analyst and chartist at a top-tier quantitative "
        "trading firm with deep expertise in price action, momentum, and "
        "mean-reversion strategies.\n\n"
        "ANALYSIS FRAMEWORK:\n"
        "1. Trend Direction: Use SMA 20/50 crossovers and price position "
        "relative to moving averages. Price > SMA50 > SMA200 = uptrend.\n"
        "2. Momentum: RSI >70 = overbought (potential reversal), <30 = "
        "oversold (potential bounce). MACD crossover confirms momentum shift.\n"
        "3. Volatility: Bollinger Band width indicates volatility regime. "
        "Price touching upper band in uptrend = continuation; in range = "
        "reversal. ATR measures daily volatility for stop-loss sizing.\n"
        "4. Volume Confirmation: Price moves on >1.5x average volume are "
        "more reliable. Divergence (price up, volume down) is bearish.\n"
        "5. Support/Resistance: 52-week high/low, round numbers, and "
        "previous consolidation zones.\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"signal": "BUY", "confidence": 74, "summary": '
        '"AAPL is in a confirmed uptrend with price ($185) above both '
        "SMA20 ($182) and SMA50 ($178). RSI at 58 shows healthy momentum "
        "without overbought conditions. MACD is positive (0.85) and above "
        "signal line (0.62), confirming bullish momentum. Volume trend at "
        "1.2x average provides moderate confirmation. Bollinger Bands "
        "($180-$190) show price in upper half, consistent with uptrend. "
        "Key support at $178 (SMA50), resistance at $192 (52-week high). "
        'ATR of 2.8 suggests a stop-loss at $181 (1.5x ATR)."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
        '"confidence": <0-100>, "summary": "<your detailed technical analysis>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        prompt = self._build_prompt(ticker, context)
        response = self.llm.generate(prompt, self.system_prompt)
        result = self._parse_response(ticker, response)
        result.details = {
            "indicators": context.get("indicators", {}),
            "signals": context.get("signals", {}),
        }
        return result

    def _build_prompt(self, ticker: str, context: dict) -> str:
        indicators = context.get("indicators", {})
        signals = context.get("signals", {})

        parts = [f"Analyze the technical indicators for {ticker}.", "\nKey Indicators:"]

        if indicators.get("current_price"):
            parts.append(f"Current Price: ${indicators['current_price']:.2f}")
        if indicators.get("price_change_pct") is not None:
            parts.append(f"Period Change: {indicators['price_change_pct']:.2f}%")
        if indicators.get("rsi") is not None:
            parts.append(f"RSI (14): {indicators['rsi']:.2f}")
        if indicators.get("macd") is not None:
            parts.append(f"MACD: {indicators['macd']:.4f}")
        if indicators.get("macd_signal") is not None:
            parts.append(f"MACD Signal: {indicators['macd_signal']:.4f}")
        if indicators.get("sma_20") is not None:
            parts.append(f"SMA 20: ${indicators['sma_20']:.2f}")
        if indicators.get("sma_50") is not None:
            parts.append(f"SMA 50: ${indicators['sma_50']:.2f}")
        if indicators.get("bb_upper") is not None:
            parts.append(
                f"Bollinger Bands: ${indicators['bb_lower']:.2f} - ${indicators['bb_upper']:.2f}"
            )
        if indicators.get("atr") is not None:
            parts.append(f"ATR (14): {indicators['atr']:.4f}")
        if indicators.get("volume_trend") is not None:
            parts.append(f"Volume Trend: {indicators['volume_trend']:.2f}x average")

        if signals:
            parts.append("\nSignal Summary:")
            for key, value in signals.items():
                if key not in ("overall", "confidence"):
                    parts.append(f"- {key}: {value}")

        parts.append(
            "\nBased on these technical indicators, respond with JSON: "
            '{"signal": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL", '
            '"confidence": <0-100>, "summary": "<your analysis>"}'
        )
        return "\n".join(parts)
