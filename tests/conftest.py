from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from opentrade_ai.config import AppConfig, LLMConfig, TradingConfig
from opentrade_ai.llm.provider import LLMProvider


@pytest.fixture
def sample_price_data():
    dates = pd.date_range(start="2025-01-01", periods=60, freq="B")
    np.random.seed(42)
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + abs(np.random.normal(0.005, 0.005, len(dates)))),
            "Low": prices * (1 - abs(np.random.normal(0.005, 0.005, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_stock_info():
    return {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_000_000_000_000,
        "pe_ratio": 28.5,
        "forward_pe": 25.2,
        "dividend_yield": 0.005,
        "beta": 1.2,
        "52_week_high": 200.0,
        "52_week_low": 140.0,
        "avg_volume": 50_000_000,
        "current_price": 185.0,
        "revenue_growth": 0.08,
        "profit_margins": 0.25,
        "debt_to_equity": 1.5,
        "return_on_equity": 0.15,
        "free_cash_flow": 100_000_000_000,
    }


@pytest.fixture
def sample_news():
    return [
        {
            "title": "Apple Reports Record Q4 Earnings",
            "publisher": "Reuters",
            "published": "2025-01-15",
            "summary": "Apple beats expectations with strong iPhone sales.",
        },
        {
            "title": "New AI Features Coming to iPhone",
            "publisher": "Bloomberg",
            "published": "2025-01-14",
            "summary": "Apple announces new AI capabilities for next iOS.",
        },
    ]


@pytest.fixture
def test_llm_config():
    return LLMConfig(provider="ollama", model="llama3", ollama_base_url="http://localhost:11434")


@pytest.fixture
def test_trading_config():
    return TradingConfig(
        max_debate_rounds=1,
        risk_tolerance="moderate",
        analysis_period_days=90,
        tickers=["AAPL"],
    )


@pytest.fixture
def test_app_config(test_llm_config, test_trading_config):
    return AppConfig(llm=test_llm_config, trading=test_trading_config)


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMProvider)
    llm.generate.return_value = (
        "Based on the analysis, the stock shows strong fundamentals with solid growth. "
        "Signal: BUY with 70% confidence. The company has strong revenue growth "
        "and healthy margins. Recommend a moderate position size."
    )
    llm.is_available.return_value = True
    return llm
