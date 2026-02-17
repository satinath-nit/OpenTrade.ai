# OpenTrade.ai

Multi-agent AI stock trading advisor powered by LangGraph, with both a Streamlit visual dashboard and a CLI interface. Uses a local LLM (Ollama or LM Studio) and free stock data from Yahoo Finance.

## Architecture

8 specialized AI agents work together in a LangGraph StateGraph pipeline:

1. **Fundamental Analyst** - Evaluates company financials, valuation, growth
2. **Sentiment Analyst** - Analyzes market sentiment from news and social signals
3. **News Analyst** - Processes recent news for trading implications
4. **Technical Analyst** - Computes RSI, MACD, Bollinger Bands, SMA, EMA, ATR, etc.
5. **Bull Researcher** - Builds the strongest bullish case
6. **Bear Researcher** - Builds the strongest bearish case
7. **Trader Agent** - Synthesizes all inputs into a trading decision
8. **Risk Manager** - Reviews and can approve/modify/reject the trade

```
Fetch Data --> [Fundamental, Sentiment, News, Technical] (parallel)
                          |
                    Research Debate (Bull vs Bear)
                          |
                    Trader Decision
                          |
                      Risk Review
                          |
                    Final Decision
```

## Prerequisites

- **Python 3.10+**
- **One of the following LLM providers:**
  - [Ollama](https://ollama.com) (free, local)
  - [LM Studio](https://lmstudio.ai) (free, local, supports any GGUF model)
  - [OpenAI](https://platform.openai.com) (cloud, requires API key)

## Local Setup Instructions

### 1. Install a Local LLM Provider

**Option A: Ollama** (recommended for beginners)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - download from https://ollama.com/download

# Start Ollama and pull a model
ollama serve
ollama pull llama3    # or: ollama pull mistral
```

**Option B: LM Studio** (recommended for flexibility)

1. Download from https://lmstudio.ai
2. Open LM Studio and download any model (e.g., Llama 3, Mistral, Phi-3, etc.)
3. Go to the **Local Server** tab and click **Start Server**
4. The server runs at `http://localhost:1234/v1` by default (OpenAI-compatible API)

### 3. Set Up the Project

```bash
# Navigate to the project directory
cd opentrade-ai

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install the project and dependencies
pip install -e ".[dev]"
```

### 4. Configure Environment Variables (Optional)

```bash
# Copy the example env file
cp .env.example .env

# Edit .env if needed (defaults work for Ollama)
# The default config uses Ollama at http://localhost:11434 with llama3 model
```

### 5. Verify Setup

```bash
# Run the test suite (96 tests)
pytest tests/ -v

# Run linting
ruff check src/ tests/ cli/ app/

# Check Ollama is running
curl http://localhost:11434/api/tags
```

## Running the Application

### Streamlit Dashboard (Visual UI)

```bash
streamlit run app/streamlit_app.py
```

This opens a browser with:
- Sidebar for configuration (ticker, LLM settings, risk tolerance)
- Candlestick chart with technical indicator overlays
- Agent signals and confidence visualization
- Bull vs Bear debate view
- Risk assessment panel
- Step-by-step analysis log

### CLI Interface

```bash
# Interactive mode (prompts for ticker)
python -m cli.main

# With options
python -m cli.main --ticker AAPL --date 2025-02-15
python -m cli.main -t MSFT -p ollama -m llama3 -r moderate

# See all options
python -m cli.main --help
```

**CLI Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-t, --ticker` | Stock ticker symbol | (interactive prompt) |
| `-d, --date` | Analysis date (YYYY-MM-DD) | today |
| `-p, --provider` | LLM provider (ollama/lmstudio/openai) | ollama |
| `-m, --model` | LLM model name | llama3 |
| `-r, --risk` | Risk tolerance (conservative/moderate/aggressive) | moderate |

## Project Structure

```
opentrade-ai/
├── app/
│   └── streamlit_app.py          # Streamlit visual dashboard
├── cli/
│   ├── __init__.py
│   └── main.py                   # Rich CLI interface
├── src/opentrade_ai/
│   ├── agents/
│   │   ├── base.py               # BaseAgent + AnalysisResult
│   │   ├── fundamental_analyst.py
│   │   ├── sentiment_analyst.py
│   │   ├── news_analyst.py
│   │   ├── technical_analyst.py
│   │   ├── bull_researcher.py
│   │   ├── bear_researcher.py
│   │   ├── trader.py
│   │   └── risk_manager.py
│   ├── analysis/
│   │   └── technical_indicators.py  # RSI, MACD, BB, SMA, EMA, ATR, etc.
│   ├── data/
│   │   └── market_data.py        # Yahoo Finance data provider
│   ├── graph/
│   │   └── trading_graph.py      # LangGraph StateGraph orchestration
│   ├── llm/
│   │   └── provider.py           # Ollama/LM Studio/OpenAI LLM provider
│   └── config.py                 # Application configuration
├── tests/                        # 96 tests covering all modules
│   ├── conftest.py
│   ├── test_agents.py
│   ├── test_config.py
│   ├── test_llm_provider.py
│   ├── test_market_data.py
│   ├── test_technical_indicators.py
│   └── test_trading_graph.py
├── .env.example                  # Environment variable template
├── .gitignore
├── pyproject.toml                # Project config + dependencies
└── requirements.txt              # Pinned dependencies
```

## Using with LM Studio

```bash
# Make sure LM Studio server is running, then:
python -m cli.main -t AAPL -p lmstudio -m local-model

# Or via Streamlit (select "lmstudio" from the provider dropdown)
streamlit run app/streamlit_app.py
```

LM Studio exposes an OpenAI-compatible API at `http://localhost:1234/v1`. You can change the URL in `.env` via `LMSTUDIO_BASE_URL` if your server runs on a different port.

## Using with OpenAI

```bash
# Set your API key in .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
echo "LLM_PROVIDER=openai" >> .env

# Run with OpenAI
python -m cli.main -t AAPL -p openai -m gpt-4o-mini
```

## Technical Indicators Computed

| Indicator | Description |
|-----------|-------------|
| RSI (14) | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands (20,2) | Upper/Lower price bands |
| SMA 20/50 | Simple Moving Averages |
| EMA 12/26 | Exponential Moving Averages |
| ATR (14) | Average True Range (volatility) |
| Stochastic (14,3) | Stochastic Oscillator |
| OBV | On-Balance Volume |

## Documentation

Detailed documentation is in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, agent pipeline, state management, Mermaid diagrams |
| [Usage Guide](docs/usage.md) | Installation, Streamlit dashboard, CLI, data sources, troubleshooting |
| [Configuration](docs/configuration.md) | All environment variables, recommended settings per use case |

## Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always do your own research and consult a financial advisor before making investment decisions. Past performance does not guarantee future results.
