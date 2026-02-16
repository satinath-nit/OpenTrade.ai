# Architecture

## System Overview

OpenTrade.ai is a multi-agent stock analysis system orchestrated via LangGraph. Nine specialized AI agents collaborate through a directed acyclic graph (DAG) to produce a final trading recommendation with confidence scoring and risk management.

## High-Level Pipeline

```mermaid
flowchart TD
    START([Start]) --> FD[Fetch Market Data]
    FD --> AN[Run Analysts<br/><i>parallel, configurable</i>]
    AN --> RD[Research Debate<br/><i>Bull vs Bear</i>]
    RD --> TD[Trader Decision]
    TD --> RR[Risk Review]
    RR --> VR[Verification]
    VR --> END([Final Decision])

    subgraph "Data Sources"
        YF[Yahoo Finance<br/>Price, Financials, News]
        GN[Google News<br/>Web Articles]
        SE[SEC EDGAR<br/>10-K, 10-Q, 8-K]
        GT[Google Trends<br/>Search Interest]
    end

    FD --> YF
    FD --> GN
    FD --> SE
    FD --> GT
```

## Agent Architecture

```mermaid
flowchart LR
    subgraph Analysts ["Analyst Agents (parallel)"]
        FA[Fundamental<br/>Analyst]
        SA[Sentiment<br/>Analyst]
        NA[News<br/>Analyst]
        TA[Technical<br/>Analyst]
    end

    subgraph Debate ["Research Debate"]
        BR[Bull<br/>Researcher]
        BeR[Bear<br/>Researcher]
        BR <-->|debate rounds| BeR
    end

    subgraph Decision ["Decision Layer"]
        TR[Trader<br/>Agent]
        RM[Risk<br/>Manager]
        VR[Verifier<br/>Agent]
    end

    Analysts --> Debate
    Debate --> Decision
```

## State Management

The system uses LangGraph's `StateGraph` with a `TypedDict` state object. State flows through each node, accumulating analysis results.

```mermaid
stateDiagram-v2
    [*] --> FetchData
    FetchData --> RunAnalysts : stock_info, news, indicators, signals
    RunAnalysts --> ResearchDebate : analyst_reports[]
    ResearchDebate --> TraderDecision : debate_history[]
    TraderDecision --> RiskReview : trader_signal, trader_confidence
    RiskReview --> Verification : final_signal, final_confidence
    Verification --> [*] : TradingDecision
```

### State Fields

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `str` | Stock ticker symbol |
| `date` | `str \| None` | Analysis date |
| `stock_info` | `dict` | Company fundamentals from Yahoo Finance |
| `news` | `list` | Yahoo Finance news articles |
| `google_news` | `list` | Google News articles |
| `sec_filings` | `list` | SEC EDGAR filings |
| `google_trends` | `dict` | Google Trends interest data |
| `price_data` | `DataFrame` | Historical OHLCV price data |
| `indicators` | `dict` | Computed technical indicators |
| `signals` | `dict` | Technical signal summary |
| `analyst_reports` | `list[AnalysisResult]` | All analyst outputs |
| `debate_history` | `list[dict]` | Bull/Bear debate rounds |
| `trader_summary` | `str` | Trader's analysis |
| `risk_assessment` | `str` | Risk manager's review |
| `final_signal` | `str` | Final recommendation |
| `final_confidence` | `float` | Confidence percentage |

## Agent Details

### 1. Fundamental Analyst

Evaluates company financials, valuation metrics, and growth trajectory. Consumes `stock_info`, `indicators`, `signals`, and `sec_filings`.

### 2. Sentiment Analyst

Analyzes market sentiment from news coverage and social signals. Consumes `stock_info`, `news`, `google_news`, and `google_trends`.

### 3. News Analyst

Processes recent news for trading-relevant implications, catalysts, and risks. Consumes `stock_info`, `news`, `google_news`, and `sec_filings`.

### 4. Technical Analyst

Interprets quantitative indicators (RSI, MACD, Bollinger Bands, moving averages, etc.). Consumes `indicators` and `signals`.

### 5. Bull Researcher

Builds the strongest possible bullish case using all analyst reports. Engages in multi-round debate with the Bear Researcher.

### 6. Bear Researcher

Builds the strongest possible bearish case. Counter-argues the Bull Researcher across debate rounds.

### 7. Trader Agent

Synthesizes all analyst reports, debate outcomes, and risk tolerance into a final trading decision with signal and confidence.

### 8. Risk Manager

Reviews the trader's decision against risk parameters. Can approve, modify (reduce confidence), or reject (force hold) the trade.

### 9. Verifier Agent

Quality gate that checks for contradictions, unsupported claims, and internal consistency across all analysis outputs.

## Parallel Execution

Analyst agents run in parallel using `ThreadPoolExecutor`. The number of concurrent agents is configurable via `MAX_PARALLEL_AGENTS` (default: 2).

```mermaid
sequenceDiagram
    participant G as Graph
    participant TP as ThreadPool(max_workers=N)
    participant FA as Fundamental
    participant SA as Sentiment
    participant NA as News
    participant TA as Technical

    G->>TP: submit all 4 analysts
    par Batch 1 (workers=2)
        TP->>FA: analyze()
        TP->>SA: analyze()
    end
    FA-->>TP: result
    par Batch 2
        TP->>NA: analyze()
    end
    SA-->>TP: result
    NA-->>TP: result
    TP->>TA: analyze()
    TA-->>TP: result
    TP-->>G: all 4 results
```

## LLM Integration

```mermaid
flowchart TD
    Agent[Any Agent] --> LLM[LLMProvider]
    LLM --> |provider=ollama| OL[Ollama<br/>localhost:11434]
    LLM --> |provider=lmstudio| LS[LM Studio<br/>localhost:1234/v1]
    LLM --> |provider=openai| OA[OpenAI API]

    OL --> |llama3, mistral, etc.| Model[Local Model]
    LS --> |any GGUF model| Model
    OA --> |gpt-4o, gpt-4o-mini| Cloud[Cloud Model]
```

All LLM calls include retry logic (3 attempts, 2-second delay) to handle transient failures common with local LLMs.

## Data Flow

```mermaid
flowchart TD
    subgraph Fetch ["1. Fetch Market Data"]
        YF[yfinance] --> Price[Price Data<br/>OHLCV]
        YF --> Info[Stock Info<br/>Fundamentals]
        YF --> YNews[Yahoo News]
        GN[gnews] --> GNews[Google News<br/>Articles + URLs]
        SEC[SEC EDGAR API] --> Filings[10-K, 10-Q, 8-K<br/>Filings + URLs]
        GT[pytrends] --> Trends[Search Interest<br/>Trend Direction]
        Price --> TI[Technical Analyzer]
        TI --> Indicators[RSI, MACD, BB<br/>SMA, EMA, ATR<br/>Stochastic, OBV]
    end

    subgraph Analyze ["2. Analyst Layer"]
        Info --> FA[Fundamental]
        Filings --> FA
        YNews --> SA[Sentiment]
        GNews --> SA
        Trends --> SA
        YNews --> NA[News]
        GNews --> NA
        Filings --> NA
        Indicators --> TA[Technical]
    end

    subgraph Synthesize ["3. Decision Layer"]
        FA & SA & NA & TA --> Bull & Bear
        Bull & Bear --> Trader
        Trader --> Risk
        Risk --> Verifier
    end

    Verifier --> Output[TradingDecision<br/>signal + confidence]
```

## Module Structure

```
src/opentrade_ai/
    agents/
        base.py                  # BaseAgent, AnalysisResult, AgentRole
        fundamental_analyst.py   # Company financials analysis
        sentiment_analyst.py     # Market sentiment analysis
        news_analyst.py          # News impact analysis
        technical_analyst.py     # Technical indicators analysis
        bull_researcher.py       # Bullish case builder
        bear_researcher.py       # Bearish case builder
        trader.py                # Trading decision maker
        risk_manager.py          # Risk assessment
        verifier.py              # Quality verification
    analysis/
        technical_indicators.py  # RSI, MACD, BB, SMA, EMA, ATR, etc.
    data/
        market_data.py           # Unified data provider
        google_news.py           # Google News integration
        sec_edgar.py             # SEC EDGAR API client
        google_trends.py         # Google Trends integration
    graph/
        trading_graph.py         # LangGraph StateGraph orchestration
    llm/
        provider.py              # Multi-provider LLM client
    config.py                    # Configuration management
    report.py                    # JSON/HTML report export
    screener.py                  # Multi-ticker screener
```
