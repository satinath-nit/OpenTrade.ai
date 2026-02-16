import numpy as np
import pandas as pd
import pytest

from opentrade_ai.analysis.technical_indicators import TechnicalAnalyzer


class TestTechnicalAnalyzer:
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_compute_indicators_returns_dict(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        assert isinstance(result, dict)

    def test_compute_indicators_has_required_keys(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        required_keys = ["rsi", "current_price", "price_change_pct", "avg_volume", "volume_trend"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_compute_indicators_rsi_in_range(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        if result["rsi"] is not None:
            assert 0 <= result["rsi"] <= 100

    def test_compute_indicators_current_price_positive(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        assert result["current_price"] > 0

    def test_compute_indicators_insufficient_data(self):
        small_df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [103, 104, 105],
                "Low": [98, 99, 100],
                "Close": [101, 102, 103],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2025-01-01", periods=3),
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            self.analyzer.compute_indicators(small_df)

    def test_compute_indicators_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Insufficient data"):
            self.analyzer.compute_indicators(empty_df)

    def test_compute_indicators_has_macd(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        assert "macd" in result
        assert "macd_signal" in result

    def test_compute_indicators_has_bollinger_bands(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        assert "bb_upper" in result
        assert "bb_lower" in result

    def test_compute_indicators_has_moving_averages(self, sample_price_data):
        result = self.analyzer.compute_indicators(sample_price_data)
        assert "sma_20" in result
        assert "ema_12" in result


class TestSignalSummary:
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_signal_summary_returns_dict(self):
        indicators = {"rsi": 45, "macd": 0.5, "macd_signal": 0.3, "current_price": 150}
        result = self.analyzer.get_signal_summary(indicators)
        assert isinstance(result, dict)

    def test_rsi_oversold_signal(self):
        indicators = {"rsi": 25}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bullish" in result.get("rsi", "").lower()

    def test_rsi_overbought_signal(self):
        indicators = {"rsi": 75}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bearish" in result.get("rsi", "").lower()

    def test_rsi_neutral_signal(self):
        indicators = {"rsi": 50}
        result = self.analyzer.get_signal_summary(indicators)
        assert "neutral" in result.get("rsi", "").lower()

    def test_macd_bullish_crossover(self):
        indicators = {"macd": 1.0, "macd_signal": 0.5}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bullish" in result.get("macd", "").lower()

    def test_macd_bearish_crossover(self):
        indicators = {"macd": 0.3, "macd_signal": 0.8}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bearish" in result.get("macd", "").lower()

    def test_overall_signal_present(self, sample_price_data):
        indicators = self.analyzer.compute_indicators(sample_price_data)
        result = self.analyzer.get_signal_summary(indicators)
        assert "overall" in result
        assert result["overall"] in ("bullish", "bearish", "neutral", "insufficient data")

    def test_confidence_present(self, sample_price_data):
        indicators = self.analyzer.compute_indicators(sample_price_data)
        result = self.analyzer.get_signal_summary(indicators)
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 100

    def test_high_volume_signal(self):
        indicators = {"volume_trend": 2.0}
        result = self.analyzer.get_signal_summary(indicators)
        assert "high volume" in result.get("volume", "").lower()

    def test_sma_bullish(self):
        indicators = {"current_price": 160, "sma_20": 150}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bullish" in result.get("sma_20", "").lower()

    def test_sma_bearish(self):
        indicators = {"current_price": 140, "sma_20": 150}
        result = self.analyzer.get_signal_summary(indicators)
        assert "bearish" in result.get("sma_20", "").lower()


class TestSafeLast:
    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_safe_last_with_valid_series(self):
        series = pd.Series([1.0, 2.0, 3.0])
        assert self.analyzer._safe_last(series) == 3.0

    def test_safe_last_with_none(self):
        assert self.analyzer._safe_last(None) is None

    def test_safe_last_with_empty_series(self):
        assert self.analyzer._safe_last(pd.Series(dtype=float)) is None

    def test_safe_last_with_nan(self):
        series = pd.Series([1.0, np.nan])
        assert self.analyzer._safe_last(series) is None
