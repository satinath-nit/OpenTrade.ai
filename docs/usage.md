# Usage Guide

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **A local LLM provider** (choose one):
   - [Ollama](https://ollama.com) — free, local, easiest setup
   - [LM Studio](https://lmstudio.ai) — free, local, supports any GGUF model
   - [OpenAI](https://platform.openai.com) — cloud, requires API key

### Installation

```bash
cd opentrade-ai
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

pip install -e ".[dev]"
cp .env.example .env        # edit with your settings
```

### Setting Up Your LLM

**Ollama:**

```bash
# Install
brew install ollama          # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Start and pull a model
ollama serve
ollama pull llama3
```

**LM Studio:**

1. Download from https://lmstudio.ai
2. Open LM Studio, download any model (Llama 3, Mistral, Phi-3, etc.)
3. Go to **Local Server** tab, click **Start Server**
4. Server runs at `http://localhost:1234/v1` (OpenAI-compatible)

**OpenAI:**

```bash
echo "LLM_PROVIDER=openai" >> .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

---

## Streamlit Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Opens a browser at `http://localhost:8501` with:

### Sidebar Configuration

| Setting | Description |
|---------|-------------|
| **Stock Ticker** | Enter any valid ticker (e.g., AAPL, MSFT, NVDA) |
| **Analysis Date** | Date for the analysis (defaults to today) |
| **LLM Provider** | ollama, lmstudio, or openai |
| **Model Name** | Model to use (e.g., llama3, local-model, gpt-4o-mini) |
| **Risk Tolerance** | conservative, moderate, or aggressive |
| **Max Debate Rounds** | Number of bull/bear debate rounds (1-5) |
| **Analysis Period** | Days of historical data to analyze (30-365) |

### Tabs

| Tab | Content |
|-----|---------|
| **Chart** | Candlestick chart with technical indicator overlays (RSI gauge, MACD, Bollinger Bands, volume) |
| **Agents** | Signal distribution chart + expandable summaries per analyst |
| **Debate** | Bull vs Bear arguments per debate round |
| **Risk** | Risk manager's assessment and any adjustments |
| **Verification** | Verifier's quality gate results and issues found |
| **Log** | Detailed pipeline log with data source citations, links, and per-step metrics |

### Modes

**Single Stock Analysis** — Analyze one ticker in depth.

**Stock Screener** — Enter a watchlist of tickers, rank the top N by trading opportunity. Results show ranked picks with signals, confidence, rationale, and position sizing.

### Report Export

After analysis, download reports via buttons:
- **JSON Report** — machine-readable full analysis data
- **HTML Report** — styled, shareable report

---

## CLI Interface

```bash
# Interactive mode
python -m cli.main

# With options
python -m cli.main --ticker AAPL --date 2026-02-15
python -m cli.main -t MSFT -p lmstudio -m local-model -r moderate

# Screener mode
python -m cli.main --screener -w "AAPL,MSFT,GOOGL,NVDA,AMZN" --top-n 5

# Help
python -m cli.main --help
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --ticker` | Stock ticker symbol | (interactive prompt) |
| `-d, --date` | Analysis date (YYYY-MM-DD) | today |
| `-p, --provider` | LLM provider (ollama/lmstudio/openai) | ollama |
| `-m, --model` | LLM model name | llama3 |
| `-r, --risk` | Risk tolerance | moderate |
| `--screener` | Enable screener mode | false |
| `-w, --watchlist` | Comma-separated tickers for screener | — |
| `--top-n` | Number of top picks to show | 5 |

---

## Data Sources

All data sources are **free** and configurable. Toggle them in `.env`:

| Source | Library | API Key | Toggle |
|--------|---------|---------|--------|
| Yahoo Finance | `yfinance` | None | Always on |
| Google News | `gnews` | None | `ENABLE_GOOGLE_NEWS` |
| SEC EDGAR | `requests` | None | `ENABLE_SEC_EDGAR` |
| Google Trends | `pytrends` | None | `ENABLE_GOOGLE_TRENDS` |

### Yahoo Finance (always enabled)

Provides historical OHLCV prices, company fundamentals, and news. Data from https://finance.yahoo.com.

### Google News

Broader web news coverage via Google News. Searches for `"{company_name} stock"` articles. Configure period and max results in `.env`.

### SEC EDGAR

Public SEC filings (10-K, 10-Q, 8-K) from https://www.sec.gov/edgar. Free API with 10 req/sec rate limit. No key needed.

### Google Trends

Search interest trends from https://trends.google.com. Shows whether public interest in the stock is rising, stable, or declining.

---

## Technical Indicators

The system computes these indicators from historical price data:

| Indicator | Period | Description |
|-----------|--------|-------------|
| RSI | 14 | Relative Strength Index (oversold < 30, overbought > 70) |
| MACD | 12/26/9 | Moving Average Convergence Divergence |
| Bollinger Bands | 20, 2 | Upper/lower price bands (2 std deviations) |
| SMA 20 | 20 | Short-term Simple Moving Average |
| SMA 50 | 50 | Medium-term Simple Moving Average |
| EMA 12 | 12 | Short-term Exponential Moving Average |
| EMA 26 | 26 | Medium-term Exponential Moving Average |
| ATR | 14 | Average True Range (volatility measure) |
| Stochastic | 14, 3 | Stochastic Oscillator (%K, %D) |
| OBV | — | On-Balance Volume |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific test file
pytest tests/test_trading_graph.py -v

# With coverage
pytest tests/ --cov=opentrade_ai --cov-report=html

# Linting
ruff check src/ tests/ cli/ app/
```

---

## Troubleshooting

### "Cannot connect to Ollama"

Ollama isn't running. Start it:

```bash
ollama serve
```

### "ModuleNotFoundError: No module named 'pandas_ta'"

Install the project in your venv:

```bash
source venv/bin/activate
pip install -e ".[dev]"
```

### Streamlit uses system Python instead of venv

Use `python -m streamlit` instead of `streamlit`:

```bash
python -m streamlit run app/streamlit_app.py
```

### Analysts show 0% confidence

Your local LLM may be overwhelmed by parallel requests. Reduce parallelism:

```bash
# In .env
MAX_PARALLEL_AGENTS=1
```

### LM Studio not working with Streamlit

Make sure the Streamlit sidebar **LLM Provider** dropdown is set to `lmstudio`. Or set it in `.env`:

```bash
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```
