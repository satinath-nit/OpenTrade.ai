# Configuration Reference

All configuration is managed through environment variables (`.env` file) and/or Streamlit sidebar controls. Sidebar values override `.env` defaults.

## LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `lmstudio`, or `openai` |
| `OLLAMA_MODEL` | `llama3` | Model name for Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint (OpenAI-compatible) |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key (required for `openai` provider) |

### Provider Details

**Ollama** (`LLM_PROVIDER=ollama`)
- Runs models locally via [Ollama](https://ollama.com)
- Endpoint: `http://localhost:11434/api/generate`
- Popular models: `llama3`, `mistral`, `phi3`, `gemma`

**LM Studio** (`LLM_PROVIDER=lmstudio`)
- Runs any GGUF model locally via [LM Studio](https://lmstudio.ai)
- Exposes an OpenAI-compatible API at `http://localhost:1234/v1`
- Set model name to match what LM Studio reports, or use `local-model`

**OpenAI** (`LLM_PROVIDER=openai`)
- Uses OpenAI's cloud API
- Requires `OPENAI_API_KEY`
- Popular models: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

## Trading Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL_AGENTS` | `2` | Number of analyst agents to run in parallel (1-4). `1` = sequential (safest for slow machines), `4` = full parallel (best for cloud APIs or powerful hardware) |

These settings are also configurable via the Streamlit sidebar:

| Setting | Default | Range |
|---------|---------|-------|
| Risk Tolerance | `moderate` | `conservative`, `moderate`, `aggressive` |
| Max Debate Rounds | `2` | 1-5 |
| Analysis Period (days) | `90` | 30-365 |

## Data Source Configuration

All data sources are free. Toggle them on/off via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_GOOGLE_NEWS` | `true` | Enable Google News article fetching |
| `ENABLE_SEC_EDGAR` | `true` | Enable SEC EDGAR filing retrieval |
| `ENABLE_GOOGLE_TRENDS` | `true` | Enable Google Trends interest data |
| `GOOGLE_NEWS_PERIOD` | `7d` | Google News lookback period (`1d`, `7d`, `30d`) |
| `GOOGLE_NEWS_MAX_RESULTS` | `10` | Max Google News articles per query |
| `SEC_EDGAR_MAX_FILINGS` | `5` | Max SEC filings to retrieve per ticker |
| `GOOGLE_TRENDS_TIMEFRAME` | `today 3-m` | Google Trends timeframe (`today 1-m`, `today 3-m`, `today 12-m`) |

### Disabling a Data Source

If a data source is causing issues (network errors, rate limits, etc.), disable it:

```bash
ENABLE_GOOGLE_NEWS=false
ENABLE_SEC_EDGAR=false
ENABLE_GOOGLE_TRENDS=false
```

The system gracefully degrades — agents will work with whatever data is available.

## Example `.env` File

```bash
# === LLM Provider ===
LLM_PROVIDER=lmstudio
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=

# === Parallel Execution ===
MAX_PARALLEL_AGENTS=2

# === Data Sources ===
ENABLE_GOOGLE_NEWS=true
ENABLE_SEC_EDGAR=true
ENABLE_GOOGLE_TRENDS=true
GOOGLE_NEWS_PERIOD=7d
GOOGLE_NEWS_MAX_RESULTS=10
SEC_EDGAR_MAX_FILINGS=5
GOOGLE_TRENDS_TIMEFRAME=today 3-m
```

## Recommended Settings

### For Local LLMs (Ollama / LM Studio)

```bash
LLM_PROVIDER=lmstudio        # or ollama
MAX_PARALLEL_AGENTS=2         # balance speed vs stability
ENABLE_GOOGLE_TRENDS=false    # pytrends can be flaky
```

### For Cloud APIs (OpenAI)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MAX_PARALLEL_AGENTS=4         # cloud APIs handle concurrency well
```

### For Slow Machines / Debugging

```bash
MAX_PARALLEL_AGENTS=1         # run analysts sequentially
ENABLE_GOOGLE_NEWS=false
ENABLE_SEC_EDGAR=false
ENABLE_GOOGLE_TRENDS=false
```
