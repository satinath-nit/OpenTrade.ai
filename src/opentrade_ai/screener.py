from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from opentrade_ai.analysis.technical_indicators import TechnicalAnalyzer
from opentrade_ai.config import AppConfig
from opentrade_ai.data.market_data import MarketDataProvider
from opentrade_ai.llm.provider import LLMProvider


def parse_watchlist_input(raw: str) -> list[str]:
    tokens = re.split(r"[,\n\s]+", raw.strip())
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


@dataclass
class ScreenerPick:
    ticker: str
    signal: str = "hold"
    confidence: float = 0.0
    rationale: str = ""
    position_size_pct: float = 0.0
    time_horizon: str = ""
    key_risks: list[str] = field(default_factory=list)
    rank: int = 0
    stock_info: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)


@dataclass
class ScreenerResult:
    picks: list[ScreenerPick] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    watchlist: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def top_n(self, n: int) -> list[ScreenerPick]:
        return self.picks[:n]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "watchlist": self.watchlist,
            "picks": [
                {
                    "rank": p.rank,
                    "ticker": p.ticker,
                    "signal": p.signal,
                    "confidence": p.confidence,
                    "rationale": p.rationale,
                    "position_size_pct": p.position_size_pct,
                    "time_horizon": p.time_horizon,
                    "key_risks": p.key_risks,
                }
                for p in self.picks
            ],
            "errors": self.errors,
        }


SCREENER_SYSTEM_PROMPT = (
    "You are a senior portfolio strategist at a top trading firm. "
    "Given market data for multiple stocks, rank them by trading opportunity. "
    "For each stock, provide a signal, confidence, rationale, suggested position size, "
    "time horizon, and key risks.\n\n"
    "You MUST respond with a JSON object in this exact format:\n"
    '{"picks": [\n'
    '  {"ticker": "AAPL", "signal": "BUY|SELL|HOLD|STRONG BUY|STRONG SELL", '
    '"confidence": 75, "rationale": "...", "position_size_pct": 3.0, '
    '"time_horizon": "day|swing|medium|long", "key_risks": ["risk1", "risk2"]}\n'
    "]}\n"
    "Rank by confidence descending. Be concise but specific in rationale."
)


class OpenTradeScreener:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMProvider | None = None,
        on_progress: callable = None,
    ):
        self.config = config
        self.llm = llm or LLMProvider(config.llm)
        self.data_provider = MarketDataProvider()
        self.tech_analyzer = TechnicalAnalyzer()
        self.on_progress = on_progress or (lambda msg: None)

    def _gather_ticker_data(self, ticker: str, date: str | None) -> dict | None:
        period = self.config.trading.analysis_period_days
        try:
            price_data = self.data_provider.get_historical_data(ticker, period, date)
            stock_info = self.data_provider.get_stock_info(ticker)
            news = self.data_provider.get_recent_news(ticker)
            indicators = self.tech_analyzer.compute_indicators(price_data)
            signals = self.tech_analyzer.get_signal_summary(indicators)
            return {
                "ticker": ticker,
                "stock_info": stock_info,
                "news": news,
                "indicators": indicators,
                "signals": signals,
            }
        except Exception:
            return None

    def screen(
        self,
        tickers: list[str],
        date: str | None = None,
        top_n: int = 10,
    ) -> ScreenerResult:
        all_data: list[dict] = []
        errors: list[str] = []

        for i, ticker in enumerate(tickers):
            try:
                self.on_progress(f"Gathering data for {ticker} ({i+1}/{len(tickers)})...")
            except Exception:
                pass

            data = self._gather_ticker_data(ticker, date)
            if data:
                all_data.append(data)
            else:
                errors.append(f"Failed to fetch data for {ticker}")

        if not all_data:
            return ScreenerResult(
                watchlist=tickers,
                errors=errors + ["No valid ticker data available"],
            )

        prompt = self._build_screener_prompt(all_data)
        response = self.llm.generate(prompt, SCREENER_SYSTEM_PROMPT)
        picks = self._parse_screener_response(response, all_data)

        picks.sort(key=lambda p: p.confidence, reverse=True)
        for i, pick in enumerate(picks):
            pick.rank = i + 1

        picks = picks[:top_n]

        return ScreenerResult(
            picks=picks,
            watchlist=tickers,
            errors=errors,
        )

    def _build_screener_prompt(self, all_data: list[dict]) -> str:
        parts = [
            f"Analyze and rank the following {len(all_data)} stocks by trading opportunity.",
            f"Risk tolerance: {self.config.trading.risk_tolerance}",
            "",
        ]

        for data in all_data:
            ticker = data["ticker"]
            info = data["stock_info"]
            ind = data["indicators"]
            signals = data["signals"]
            news = data.get("news", [])

            parts.append(f"--- {ticker} ({info.get('name', ticker)}) ---")
            parts.append(f"Sector: {info.get('sector', 'N/A')}")

            if info.get("current_price"):
                parts.append(f"Price: ${info['current_price']:.2f}")
            if info.get("market_cap"):
                parts.append(f"Market Cap: ${info['market_cap']:,.0f}")
            if info.get("pe_ratio"):
                parts.append(f"P/E: {info['pe_ratio']:.2f}")

            if ind.get("rsi") is not None:
                parts.append(f"RSI: {ind['rsi']:.1f}")
            if ind.get("macd") is not None:
                parts.append(f"MACD: {ind['macd']:.4f}")
            if ind.get("price_change_pct") is not None:
                parts.append(f"Period Change: {ind['price_change_pct']:.2f}%")

            overall = signals.get("overall", "N/A")
            parts.append(f"Technical Signal: {overall}")

            if news:
                titles = ", ".join(n.get("title", "")[:60] for n in news[:3])
                parts.append(f"Recent Headlines: {titles}")

            parts.append("")

        parts.append(
            f"Rank all {len(all_data)} stocks. Respond with JSON: "
            '{"picks": [{"ticker": "...", "signal": "...", "confidence": N, '
            '"rationale": "...", "position_size_pct": N, '
            '"time_horizon": "...", "key_risks": [...]}]}'
        )
        return "\n".join(parts)

    def _parse_screener_response(
        self, response: str, all_data: list[dict]
    ) -> list[ScreenerPick]:
        parsed = None
        text = response.strip()

        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        elif "```json" in text:
            try:
                block = text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(block)
            except (IndexError, json.JSONDecodeError):
                parsed = None

        if isinstance(parsed, dict) and "picks" in parsed:
            picks_data = parsed["picks"]
            if isinstance(picks_data, list):
                return self._picks_from_json(picks_data, all_data)

        return self._picks_from_freetext(response, all_data)

    def _picks_from_json(
        self, picks_data: list[dict], all_data: list[dict]
    ) -> list[ScreenerPick]:
        data_map = {d["ticker"]: d for d in all_data}
        picks = []
        for item in picks_data:
            ticker = str(item.get("ticker", "")).upper()
            if not ticker:
                continue
            signal = self._normalize_signal(str(item.get("signal", "hold")))
            confidence = float(item.get("confidence", 50))
            rationale = str(item.get("rationale", ""))
            position_size = float(item.get("position_size_pct", 0))
            time_horizon = str(item.get("time_horizon", ""))
            key_risks = item.get("key_risks", [])
            if not isinstance(key_risks, list):
                key_risks = [str(key_risks)]

            td = data_map.get(ticker, {})
            picks.append(
                ScreenerPick(
                    ticker=ticker,
                    signal=signal,
                    confidence=confidence,
                    rationale=rationale,
                    position_size_pct=position_size,
                    time_horizon=time_horizon,
                    key_risks=key_risks,
                    stock_info=td.get("stock_info", {}),
                    indicators=td.get("indicators", {}),
                )
            )
        return picks

    def _picks_from_freetext(
        self, response: str, all_data: list[dict]
    ) -> list[ScreenerPick]:
        picks = []
        for data in all_data:
            ticker = data["ticker"]
            signal = "hold"
            confidence = 50.0
            lower = response.lower()
            if f"{ticker.lower()}" in lower:
                if "strong buy" in lower:
                    signal = "strong_buy"
                    confidence = 85.0
                elif "buy" in lower:
                    signal = "buy"
                    confidence = 70.0
                elif "strong sell" in lower:
                    signal = "strong_sell"
                    confidence = 85.0
                elif "sell" in lower:
                    signal = "sell"
                    confidence = 70.0
                elif "hold" in lower:
                    signal = "hold"
                    confidence = 60.0

            picks.append(
                ScreenerPick(
                    ticker=ticker,
                    signal=signal,
                    confidence=confidence,
                    rationale=response[:300],
                    stock_info=data.get("stock_info", {}),
                    indicators=data.get("indicators", {}),
                )
            )
        return picks

    def _normalize_signal(self, raw: str) -> str:
        lower = raw.lower().strip()
        if "strong" in lower and "buy" in lower:
            return "strong_buy"
        if lower == "buy" or "bullish" in lower:
            return "buy"
        if "strong" in lower and "sell" in lower:
            return "strong_sell"
        if lower == "sell" or "bearish" in lower:
            return "sell"
        if lower == "hold" or "neutral" in lower:
            return "hold"
        return "hold"
