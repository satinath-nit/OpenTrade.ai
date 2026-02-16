import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    temperature: float = 0.3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )


@dataclass
class DataSourceConfig:
    enable_google_news: bool = True
    enable_sec_edgar: bool = True
    enable_google_trends: bool = True
    google_news_period: str = "7d"
    google_news_max_results: int = 10
    sec_edgar_max_filings: int = 5
    google_trends_timeframe: str = "today 3-m"

    @classmethod
    def from_env(cls) -> "DataSourceConfig":
        def _bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "")
            if not val:
                return default
            return val.lower() in ("true", "1", "yes", "on")

        return cls(
            enable_google_news=_bool("ENABLE_GOOGLE_NEWS", True),
            enable_sec_edgar=_bool("ENABLE_SEC_EDGAR", True),
            enable_google_trends=_bool("ENABLE_GOOGLE_TRENDS", True),
            google_news_period=os.getenv("GOOGLE_NEWS_PERIOD", "7d"),
            google_news_max_results=int(os.getenv("GOOGLE_NEWS_MAX_RESULTS", "10")),
            sec_edgar_max_filings=int(os.getenv("SEC_EDGAR_MAX_FILINGS", "5")),
            google_trends_timeframe=os.getenv(
                "GOOGLE_TRENDS_TIMEFRAME", "today 3-m"
            ),
        )


@dataclass
class TradingConfig:
    max_debate_rounds: int = 2
    risk_tolerance: str = "moderate"
    analysis_period_days: int = 90
    max_parallel_agents: int = 2
    data_cache_dir: str = str(Path.home() / ".opentrade_ai" / "cache")
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"])

    def validate(self) -> list[str]:
        errors = []
        if self.max_debate_rounds < 1:
            errors.append("max_debate_rounds must be >= 1")
        if self.risk_tolerance not in ("conservative", "moderate", "aggressive"):
            errors.append("risk_tolerance must be conservative, moderate, or aggressive")
        if self.analysis_period_days < 7:
            errors.append("analysis_period_days must be >= 7")
        if not self.tickers:
            errors.append("tickers list cannot be empty")
        return errors


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        trading = TradingConfig()
        raw = os.getenv("MAX_PARALLEL_AGENTS", "")
        if raw:
            try:
                trading.max_parallel_agents = max(1, int(raw))
            except ValueError:
                pass
        return cls(
            llm=LLMConfig.from_env(),
            trading=trading,
            data_sources=DataSourceConfig.from_env(),
        )

    def validate(self) -> list[str]:
        return self.trading.validate()
