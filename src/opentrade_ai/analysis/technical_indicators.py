import pandas as pd
import pandas_ta as ta


class TechnicalAnalyzer:
    def compute_indicators(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 14:
            raise ValueError("Insufficient data for technical analysis (need at least 14 rows)")

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        indicators = {}

        indicators["sma_20"] = self._safe_last(ta.sma(close, length=20))
        indicators["sma_50"] = self._safe_last(ta.sma(close, length=50))
        indicators["ema_12"] = self._safe_last(ta.ema(close, length=12))
        indicators["ema_26"] = self._safe_last(ta.ema(close, length=26))

        rsi = ta.rsi(close, length=14)
        indicators["rsi"] = self._safe_last(rsi)

        macd_result = ta.macd(close)
        if macd_result is not None and not macd_result.empty:
            indicators["macd"] = self._safe_last(macd_result.iloc[:, 0])
            indicators["macd_signal"] = self._safe_last(macd_result.iloc[:, 1])
            indicators["macd_histogram"] = self._safe_last(macd_result.iloc[:, 2])

        bbands = ta.bbands(close, length=20)
        if bbands is not None and not bbands.empty:
            indicators["bb_upper"] = self._safe_last(bbands.iloc[:, 0])
            indicators["bb_middle"] = self._safe_last(bbands.iloc[:, 1])
            indicators["bb_lower"] = self._safe_last(bbands.iloc[:, 2])

        atr = ta.atr(high, low, close, length=14)
        indicators["atr"] = self._safe_last(atr)

        stoch = ta.stoch(high, low, close)
        if stoch is not None and not stoch.empty:
            indicators["stoch_k"] = self._safe_last(stoch.iloc[:, 0])
            indicators["stoch_d"] = self._safe_last(stoch.iloc[:, 1])

        obv = ta.obv(close, volume)
        indicators["obv"] = self._safe_last(obv)

        indicators["current_price"] = float(close.iloc[-1])
        indicators["price_change_pct"] = float(
            ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
        )
        indicators["avg_volume"] = float(volume.mean())
        indicators["volume_trend"] = float(
            volume.iloc[-5:].mean() / volume.mean() if volume.mean() > 0 else 1.0
        )

        return indicators

    def get_signal_summary(self, indicators: dict) -> dict:
        signals = {}

        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < 30:
                signals["rsi"] = "oversold (bullish)"
            elif rsi > 70:
                signals["rsi"] = "overbought (bearish)"
            else:
                signals["rsi"] = "neutral"

        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals["macd"] = "bullish crossover"
            else:
                signals["macd"] = "bearish crossover"

        price = indicators.get("current_price")
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        if price is not None and sma_20 is not None:
            if price > sma_20:
                signals["sma_20"] = "price above SMA20 (bullish)"
            else:
                signals["sma_20"] = "price below SMA20 (bearish)"
        if price is not None and sma_50 is not None:
            if price > sma_50:
                signals["sma_50"] = "price above SMA50 (bullish)"
            else:
                signals["sma_50"] = "price below SMA50 (bearish)"

        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        if price is not None and bb_upper is not None and bb_lower is not None:
            if price >= bb_upper:
                signals["bollinger"] = "at upper band (potential reversal)"
            elif price <= bb_lower:
                signals["bollinger"] = "at lower band (potential bounce)"
            else:
                signals["bollinger"] = "within bands (normal)"

        volume_trend = indicators.get("volume_trend")
        if volume_trend is not None:
            if volume_trend > 1.5:
                signals["volume"] = "high volume surge"
            elif volume_trend < 0.5:
                signals["volume"] = "low volume"
            else:
                signals["volume"] = "normal volume"

        bullish_count = sum(1 for v in signals.values() if "bullish" in v or "bounce" in v)
        bearish_count = sum(1 for v in signals.values() if "bearish" in v or "reversal" in v)
        total = len(signals)

        if total > 0:
            if bullish_count > bearish_count:
                signals["overall"] = "bullish"
            elif bearish_count > bullish_count:
                signals["overall"] = "bearish"
            else:
                signals["overall"] = "neutral"
            signals["confidence"] = round(max(bullish_count, bearish_count) / total * 100, 1)
        else:
            signals["overall"] = "insufficient data"
            signals["confidence"] = 0.0

        return signals

    def _safe_last(self, series: pd.Series | None) -> float | None:
        if series is None or series.empty:
            return None
        val = series.iloc[-1]
        if pd.isna(val):
            return None
        return float(val)
