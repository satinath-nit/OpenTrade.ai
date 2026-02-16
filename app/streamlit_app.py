import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opentrade_ai.config import AppConfig  # noqa: E402
from opentrade_ai.graph.trading_graph import OpenTradeGraph  # noqa: E402
from opentrade_ai.report import ReportExporter  # noqa: E402
from opentrade_ai.screener import OpenTradeScreener, parse_watchlist_input  # noqa: E402

st.set_page_config(page_title="OpenTrade.ai", page_icon="📈", layout="wide")

SIGNAL_COLORS = {
    "strong_buy": "#00C853",
    "buy": "#4CAF50",
    "hold": "#FFC107",
    "sell": "#FF5722",
    "strong_sell": "#D50000",
    "neutral": "#9E9E9E",
}


def init_session_state():
    defaults = {
        "decision": None,
        "analysis_running": False,
        "step_log": [],
        "price_data": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar(defaults: AppConfig):
    st.sidebar.title("Configuration")

    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").strip().upper()
    date = st.sidebar.date_input("Analysis Date", value=datetime.now().date())

    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM Settings")

    provider_options = ["ollama", "lmstudio", "openai"]
    default_provider = (
        defaults.llm.provider if defaults.llm.provider in provider_options else "ollama"
    )
    provider = st.sidebar.selectbox(
        "LLM Provider",
        provider_options,
        index=provider_options.index(default_provider),
    )

    fallback_model = {
        "ollama": "llama3",
        "lmstudio": "local-model",
        "openai": "gpt-4o-mini",
    }.get(provider, "llama3")
    env_model = defaults.llm.model if provider == default_provider else ""
    model = st.sidebar.text_input("Model Name", value=env_model or fallback_model)

    temperature = st.sidebar.slider(
        "Temperature", 0.0, 1.0, float(defaults.llm.temperature), 0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Trading Settings")

    risk_options = ["conservative", "moderate", "aggressive"]
    default_risk = (
        defaults.trading.risk_tolerance
        if defaults.trading.risk_tolerance in risk_options
        else "moderate"
    )
    risk = st.sidebar.selectbox(
        "Risk Tolerance",
        risk_options,
        index=risk_options.index(default_risk),
    )

    debate_rounds = st.sidebar.slider(
        "Debate Rounds",
        1,
        5,
        int(defaults.trading.max_debate_rounds),
    )
    period_days = st.sidebar.slider(
        "Analysis Period (days)",
        30,
        365,
        int(defaults.trading.analysis_period_days),
    )

    return {
        "ticker": ticker,
        "date": date.strftime("%Y-%m-%d"),
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "risk": risk,
        "debate_rounds": debate_rounds,
        "period_days": period_days,
    }


def create_candlestick_chart(price_data, ticker, indicators=None):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "Volume", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data["Open"],
            high=price_data["High"],
            low=price_data["Low"],
            close=price_data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if indicators and indicators.get("sma_20") is not None:
        import pandas_ta as ta

        sma20 = ta.sma(price_data["Close"], length=20)
        if sma20 is not None:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=sma20,
                    name="SMA 20",
                    line=dict(color="orange", width=1),
                ),
                row=1,
                col=1,
            )

    if indicators and indicators.get("sma_50") is not None:
        import pandas_ta as ta

        sma50 = ta.sma(price_data["Close"], length=50)
        if sma50 is not None:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=sma50,
                    name="SMA 50",
                    line=dict(color="blue", width=1),
                ),
                row=1,
                col=1,
            )

    if indicators and indicators.get("bb_upper") is not None:
        import pandas_ta as ta

        bbands = ta.bbands(price_data["Close"], length=20)
        if bbands is not None and not bbands.empty:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=bbands.iloc[:, 0],
                    name="BB Upper",
                    line=dict(color="gray", width=1, dash="dot"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=bbands.iloc[:, 2],
                    name="BB Lower",
                    line=dict(color="gray", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(128,128,128,0.1)",
                ),
                row=1,
                col=1,
            )

    colors = [
        "#26A69A" if c >= o else "#EF5350"
        for o, c in zip(price_data["Open"], price_data["Close"])
    ]
    fig.add_trace(
        go.Bar(x=price_data.index, y=price_data["Volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    if indicators and indicators.get("rsi") is not None:
        import pandas_ta as ta

        rsi = ta.rsi(price_data["Close"], length=14)
        if rsi is not None:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color="purple", width=1.5),
                ),
                row=3,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_gauge_chart(value, title, min_val=0, max_val=100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": "#1E88E5"},
                "steps": [
                    {"range": [0, 30], "color": "#EF5350"},
                    {"range": [30, 70], "color": "#FFC107"},
                    {"range": [70, 100], "color": "#4CAF50"},
                ],
            },
        )
    )
    fig.update_layout(height=250, template="plotly_dark")
    return fig


def create_signal_distribution_chart(analyst_reports):
    if not analyst_reports:
        return None
    labels = []
    values_list = []
    colors = []
    for report in analyst_reports:
        role_val = report.agent_role
        role = role_val.value if hasattr(role_val, "value") else str(role_val)
        labels.append(role.replace("_", " ").title())
        values_list.append(report.confidence)
        colors.append(SIGNAL_COLORS.get(report.signal, "#9E9E9E"))

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values_list,
            marker_color=colors,
            text=[r.signal.replace("_", " ").upper() for r in analyst_reports],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Agent Signals & Confidence",
        yaxis_title="Confidence %",
        height=350,
        template="plotly_dark",
    )
    return fig


def _safe_format(val, fmt):
    if val is None:
        return "N/A"
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return str(val)


def render_stock_info(stock_info):
    cols = st.columns(4)
    metrics = [
        ("Current Price", stock_info.get("current_price"), "${:,.2f}"),
        ("Market Cap", stock_info.get("market_cap"), "${:,.0f}"),
        ("P/E Ratio", stock_info.get("pe_ratio"), "{:.2f}"),
        ("Beta", stock_info.get("beta"), "{:.2f}"),
    ]
    for i, (label, val, fmt) in enumerate(metrics):
        with cols[i]:
            st.metric(label, _safe_format(val, fmt))

    cols2 = st.columns(4)
    metrics2 = [
        ("Revenue Growth", stock_info.get("revenue_growth"), "{:.2%}"),
        ("Profit Margins", stock_info.get("profit_margins"), "{:.2%}"),
        ("Debt/Equity", stock_info.get("debt_to_equity"), "{:.2f}"),
        ("Dividend Yield", stock_info.get("dividend_yield"), "{:.2%}"),
    ]
    for i, (label, val, fmt) in enumerate(metrics2):
        with cols2[i]:
            st.metric(label, _safe_format(val, fmt))


def render_decision_badge(decision):
    signal = decision.final_signal.replace("_", " ").upper()
    color = SIGNAL_COLORS.get(decision.final_signal, "#9E9E9E")
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:10px;
                    background:linear-gradient(135deg, {color}22, {color}44);
                    border: 2px solid {color};">
            <h1 style="color:{color}; margin:0;">{signal}</h1>
            <h3 style="color:#ccc; margin:5px 0;">Confidence: {decision.confidence:.1f}%</h3>
            <p style="color:#999;">Ticker: {decision.ticker}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_screener_results(result):
    if result.errors:
        for err in result.errors:
            st.warning(err)

    if not result.picks:
        st.error("No picks available.")
        return

    for pick in result.picks:
        color = SIGNAL_COLORS.get(pick.signal, "#9E9E9E")
        signal_label = pick.signal.replace("_", " ").upper()
        risks = ", ".join(pick.key_risks[:3]) if pick.key_risks else "N/A"

        st.markdown(
            f"""<div style="background:#16213e; padding:16px; border-radius:8px;
                    margin:8px 0; border-left:4px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h3 style="margin:0; color:white;">#{pick.rank} {pick.ticker}</h3>
                        <span style="background:{color}; color:white; padding:2px 10px;
                               border-radius:4px; font-weight:bold; font-size:0.85em;">
                            {signal_label}
                        </span>
                        <span style="color:#aaa; margin-left:12px;">
                            Confidence: {pick.confidence:.0f}%
                        </span>
                    </div>
                    <div style="text-align:right; color:#aaa; font-size:0.9em;">
                        Position: {pick.position_size_pct:.1f}% |
                        Horizon: {pick.time_horizon or 'N/A'}
                    </div>
                </div>
                <p style="color:#ccc; margin:8px 0 4px;">{pick.rationale[:300]}</p>
                <p style="color:#888; font-size:0.85em; margin:0;">
                    Risks: {risks}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )


def main():
    init_session_state()

    st.title("📈 OpenTrade.ai")
    st.caption("Multi-Agent LLM Stock Trading Advisor")

    config_defaults = AppConfig.from_env()
    params = render_sidebar(config_defaults)

    mode = st.sidebar.radio("Mode", ["Single Stock", "Screener"], horizontal=True)

    if mode == "Screener":
        _run_screener_mode(config_defaults, params)
    else:
        _run_single_stock_mode(config_defaults, params)


def _build_config(params):
    config = AppConfig.from_env()
    config.llm.provider = params["provider"]
    config.llm.model = params["model"]
    config.llm.temperature = params["temperature"]
    config.trading.risk_tolerance = params["risk"]
    config.trading.max_debate_rounds = params["debate_rounds"]
    config.trading.analysis_period_days = params["period_days"]
    return config


def _run_single_stock_mode(config_defaults, params):
    run_analysis = st.sidebar.button(
        "Run Analysis", type="primary", use_container_width=True
    )

    if run_analysis:
        st.session_state.step_log = []
        st.session_state.analysis_running = True

        config = _build_config(params)

        errors = config.validate()
        if errors:
            for err in errors:
                st.error(f"Config error: {err}")
            st.session_state.analysis_running = False
            return

        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_count = [0]
            total_steps = 8

            def on_step(step):
                try:
                    if step.status == "completed":
                        step_count[0] += 1
                        progress_bar.progress(min(step_count[0] / total_steps, 1.0))
                        status_text.info(f"{step.step_name} completed")
                        st.session_state.step_log.append(
                            {"name": step.step_name, "status": "completed", "data": step.data}
                        )
                    elif step.status == "error":
                        step_count[0] += 1
                        progress_bar.progress(min(step_count[0] / total_steps, 1.0))
                        status_text.error(f"{step.step_name}: {step.error}")
                        st.session_state.step_log.append({
                            "name": step.step_name,
                            "status": "error",
                            "error": step.error,
                            "data": step.data,
                        })
                    else:
                        status_text.info(f"{step.step_name}...")
                except Exception:
                    pass

            graph = OpenTradeGraph(config, on_step=on_step)

            try:
                decision = graph.propagate(params["ticker"], params["date"])
                st.session_state.decision = decision
                st.session_state.price_data = decision.price_data
                progress_bar.progress(1.0)
                status_text.success("Analysis complete!")
            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e) or repr(e)
                st.error(f"Analysis failed ({err_type}): {err_msg}")
                with st.expander("Full error traceback"):
                    st.code(traceback.format_exc())
                st.session_state.analysis_running = False
                return

        st.session_state.analysis_running = False

    decision = st.session_state.decision
    if decision is None:
        st.info(
            "Configure your settings in the sidebar and click **Run Analysis** to get started."
        )
        st.markdown("---")
        st.subheader("Quick Start")
        st.markdown(
            """
            1. **Enter a stock ticker** (e.g., AAPL, MSFT, NVDA)
            2. **Choose your LLM provider** (Ollama for local, OpenAI for cloud)
            3. **Set risk tolerance** and analysis parameters
            4. **Click Run Analysis** to get AI-powered trading insights

            **Prerequisites:**
            - Install [Ollama](https://ollama.com) and run `ollama pull llama3`
            - Or use [LM Studio](https://lmstudio.ai) with any local model
            - Or set your OpenAI API key in `.env`
            """
        )
        return

    st.markdown("---")
    render_decision_badge(decision)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        exporter = ReportExporter()
        report_data = exporter.decision_to_dict(decision)
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report_data, indent=2, default=str),
            file_name=f"{decision.ticker}_report.json",
            mime="application/json",
        )
    with col_dl2:
        html_content = exporter._render_decision_html(report_data)
        st.download_button(
            "Download HTML Report",
            data=html_content,
            file_name=f"{decision.ticker}_report.html",
            mime="text/html",
        )

    st.markdown("---")
    if decision.stock_info:
        st.subheader(
            f"{decision.stock_info.get('name', decision.ticker)} "
            f"({decision.stock_info.get('sector', '')})"
        )
        render_stock_info(decision.stock_info)

    st.markdown("---")
    tab_chart, tab_agents, tab_debate, tab_risk, tab_verify, tab_log = st.tabs(
        ["Chart", "Agents", "Debate", "Risk", "Verification", "Log"]
    )

    with tab_chart:
        price_data = st.session_state.price_data
        if price_data is not None and not price_data.empty:
            chart = create_candlestick_chart(
                price_data, decision.ticker, decision.indicators
            )
            st.plotly_chart(chart, use_container_width=True)

            if decision.indicators:
                st.subheader("Technical Indicators")
                ind = decision.indicators
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if ind.get("rsi") is not None:
                        st.plotly_chart(
                            create_gauge_chart(ind["rsi"], "RSI"), use_container_width=True
                        )
                with col2:
                    if ind.get("current_price") and ind.get("sma_20"):
                        price_v = ind["current_price"]
                        sma_v = ind["sma_20"]
                        pct_from_sma = ((price_v - sma_v) / sma_v) * 100
                        st.metric("Price vs SMA20", f"{pct_from_sma:+.2f}%")
                    if ind.get("macd") is not None:
                        st.metric("MACD", f"{ind['macd']:.4f}")
                with col3:
                    if ind.get("atr") is not None:
                        st.metric("ATR (Volatility)", f"{ind['atr']:.4f}")
                    if ind.get("volume_trend") is not None:
                        st.metric("Volume Trend", f"{ind['volume_trend']:.2f}x avg")
                with col4:
                    if ind.get("price_change_pct") is not None:
                        st.metric("Period Change", f"{ind['price_change_pct']:+.2f}%")
                    if ind.get("bb_upper") is not None and ind.get("bb_lower") is not None:
                        st.metric(
                            "Bollinger Range",
                            f"${ind['bb_lower']:.2f} - ${ind['bb_upper']:.2f}",
                        )

                if decision.signals:
                    st.subheader("Signal Summary")
                    for key, value in decision.signals.items():
                        if key not in ("overall", "confidence"):
                            if "bullish" in value or "bounce" in value:
                                st.success(f"**{key.upper()}**: {value}")
                            elif "bearish" in value or "reversal" in value:
                                st.error(f"**{key.upper()}**: {value}")
                            else:
                                st.info(f"**{key.upper()}**: {value}")

    with tab_agents:
        if decision.analyst_reports:
            chart = create_signal_distribution_chart(decision.analyst_reports)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            for report in decision.analyst_reports:
                role_val = report.agent_role
                role = role_val.value if hasattr(role_val, "value") else str(role_val)
                with st.expander(
                    f"{role.replace('_', ' ').title()} - "
                    f"{report.signal.replace('_', ' ').upper()} ({report.confidence:.0f}%)"
                ):
                    st.markdown(report.summary)

        if decision.trader_summary:
            st.subheader("Trader Decision")
            st.markdown(decision.trader_summary)

    with tab_debate:
        if decision.debate_history:
            for i, entry in enumerate(decision.debate_history):
                st.subheader(f"Round {i + 1}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div style='border-left:4px solid #4CAF50; padding-left:12px;'>"
                        f"<h4 style='color:#4CAF50;'>Bull Case</h4>"
                        f"{entry.get('bull', 'N/A')}</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='border-left:4px solid #EF5350; padding-left:12px;'>"
                        f"<h4 style='color:#EF5350;'>Bear Case</h4>"
                        f"{entry.get('bear', 'N/A')}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("---")
        else:
            st.info("No debate history available. Run an analysis to see the bull/bear debate.")

    with tab_risk:
        if decision.risk_assessment:
            st.subheader("Risk Assessment")
            st.markdown(decision.risk_assessment)
        else:
            st.info("No risk assessment available.")

    with tab_verify:
        if decision.verification_summary:
            st.subheader("Verification Summary")
            st.markdown(decision.verification_summary)
            if decision.verification_issues:
                st.subheader("Issues Found")
                for issue in decision.verification_issues:
                    st.warning(issue)
        else:
            st.info("No verification data available.")

    with tab_log:
        st.subheader("Analysis Pipeline Log")
        _render_pipeline_log(st.session_state.step_log)


def _render_pipeline_log(step_log):
    for entry in step_log:
        name = entry["name"]
        status = entry["status"]
        data = entry.get("data", {})

        if status == "completed":
            icon = "white_check_mark"
        else:
            icon = "x"

        with st.expander(
            f":{icon}: **{name}**" if status == "completed"
            else f":{icon}: **{name}** — {entry.get('error', 'Error')}",
            expanded=(name == "Fetching Market Data"),
        ):
            if status == "error":
                st.error(entry.get("error", "Unknown error"))

            if name == "Fetching Market Data" and data:
                _render_fetch_data_log(data)
            elif name == "Research Debate" and data:
                _render_debate_log(data)
            elif name == "Trader Decision" and data:
                _render_trader_log(data)
            elif name == "Risk Review" and data:
                _render_risk_log(data)
            elif name == "Verification" and data:
                _render_verification_log(data)
            elif "signal" in data and "confidence" in data:
                st.markdown(
                    f"**Signal:** `{data['signal']}` &nbsp; "
                    f"**Confidence:** `{data['confidence']:.0f}%`"
                )
            elif not data:
                st.caption("No additional details captured for this step.")


def _render_fetch_data_log(data):
    ticker = data.get("ticker", "")
    company = data.get("company", ticker)
    sector = data.get("sector", "")
    rows = data.get("rows", 0)

    st.markdown(
        f"**{company}** ({ticker}) — {sector} &nbsp;|&nbsp; "
        f"**{rows}** days of price history loaded"
    )

    sources = data.get("data_sources", [])
    if not sources:
        st.caption("No detailed source information available.")
        return

    for src in sources:
        src_name = src.get("name", "Unknown")
        src_status = src.get("status", "unknown")
        src_url = src.get("url", "")
        details = src.get("details", "")

        if src_status == "ok":
            badge = "`OK`"
        elif src_status == "disabled":
            badge = "`DISABLED`"
        elif src_status == "no_data":
            badge = "`NO DATA`"
        else:
            badge = f"`{src_status.upper()}`"

        link = f"[{src_name}]({src_url})" if src_url else src_name
        st.markdown(f"**{link}** &nbsp; {badge} &nbsp; {details}")

        if src_name == "Yahoo Finance":
            headlines = src.get("headlines", [])
            if headlines:
                for h in headlines[:5]:
                    if h:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {h}")

        elif src_name == "Google News":
            articles = src.get("articles", [])
            if articles:
                for a in articles[:5]:
                    title = a.get("title", "")
                    url = a.get("url", "")
                    if title:
                        if url:
                            st.markdown(
                                f"&nbsp;&nbsp;&nbsp;&nbsp;- [{title}]({url})"
                            )
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {title}")

        elif src_name == "SEC EDGAR":
            filings = src.get("filings", [])
            if filings:
                for f in filings[:5]:
                    form = f.get("form", "")
                    date = f.get("date", "")
                    url = f.get("url", "")
                    label = f"{form} ({date})" if date else form
                    if url:
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;- [{label}]({url})"
                        )
                    else:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {label}")

        elif src_name == "Google Trends":
            trend = src.get("trend", "")
            avg = src.get("average_interest", 0)
            cur = src.get("current_interest", 0)
            if trend and trend not in ("disabled", "error"):
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;Trend: **{trend}** &nbsp;|&nbsp; "
                    f"Avg interest: {avg} &nbsp;|&nbsp; Current: {cur}"
                )

    indicators = data.get("indicators_computed", [])
    if indicators:
        st.markdown(
            f"**Technical indicators computed:** {', '.join(indicators)}"
        )

    signals = data.get("signals", {})
    if signals:
        overall = signals.get("overall", "")
        confidence = signals.get("confidence", "")
        if overall:
            st.markdown(
                f"**Technical signal:** `{overall}` "
                f"(confidence: {confidence})"
            )


def _render_debate_log(data):
    rounds = data.get("rounds", 0)
    bull_sig = data.get("bull_signal", "")
    bull_conf = data.get("bull_confidence", 0)
    bear_sig = data.get("bear_signal", "")
    bear_conf = data.get("bear_confidence", 0)
    st.markdown(f"**Debate rounds:** {rounds}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f":chart_with_upwards_trend: **Bull:** `{bull_sig}` ({bull_conf:.0f}%)"
        )
    with col2:
        st.markdown(
            f":chart_with_downwards_trend: **Bear:** `{bear_sig}` ({bear_conf:.0f}%)"
        )
    inputs = data.get("inputs_used", [])
    if inputs:
        st.caption("Inputs: " + ", ".join(inputs))


def _render_trader_log(data):
    signal = data.get("signal", "")
    confidence = data.get("confidence", 0)
    st.markdown(
        f"**Decision:** `{signal.upper()}` &nbsp; "
        f"**Confidence:** `{confidence:.0f}%`"
    )
    inputs = data.get("inputs_used", [])
    if inputs:
        st.caption("Inputs: " + ", ".join(inputs))


def _render_risk_log(data):
    risk_signal = data.get("risk_signal", "")
    tolerance = data.get("risk_tolerance", "")
    orig_signal = data.get("original_signal", "")
    orig_conf = data.get("original_confidence", 0)
    final_signal = data.get("final_signal", "")
    final_conf = data.get("final_confidence", 0)
    st.markdown(f"**Risk verdict:** `{risk_signal.upper()}` &nbsp; **Tolerance:** `{tolerance}`")
    if orig_signal != final_signal or abs(orig_conf - final_conf) > 0.1:
        st.markdown(
            f"Signal adjusted: `{orig_signal}` ({orig_conf:.0f}%) "
            f"→ `{final_signal}` ({final_conf:.0f}%)"
        )
    else:
        st.markdown(f"**Final:** `{final_signal}` ({final_conf:.0f}%) — no adjustment")


def _render_verification_log(data):
    verdict = data.get("verdict", "")
    issues_count = data.get("issues_count", 0)
    conf_adj = data.get("confidence_adjustment", 0)
    st.markdown(
        f"**Verdict:** `{verdict.upper()}` &nbsp; "
        f"**Issues found:** {issues_count} &nbsp; "
        f"**Confidence adjustment:** {conf_adj:+.0f}"
    )
    issues = data.get("issues", [])
    if issues:
        for issue in issues:
            st.warning(issue)
    inputs = data.get("inputs_reviewed", [])
    if inputs:
        st.caption("Reviewed: " + ", ".join(inputs))


def _run_screener_mode(config_defaults, params):
    st.subheader("Stock Screener")
    watchlist_input = st.text_area(
        "Enter tickers (comma or newline separated)",
        value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
        height=100,
    )
    top_n = st.slider("Top N picks", 1, 20, 5)

    run_screener = st.sidebar.button(
        "Run Screener", type="primary", use_container_width=True
    )

    if run_screener:
        tickers = parse_watchlist_input(watchlist_input)
        if not tickers:
            st.error("No valid tickers provided.")
            return

        config = _build_config(params)
        errors = config.validate()
        if errors:
            for err in errors:
                st.error(f"Config error: {err}")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        count = [0]

        def on_progress(msg):
            try:
                count[0] += 1
                progress_bar.progress(min(count[0] / len(tickers), 1.0))
                status_text.info(msg)
            except Exception:
                pass

        screener = OpenTradeScreener(config, on_progress=on_progress)

        try:
            result = screener.screen(tickers, params["date"], top_n)
            st.session_state["screener_result"] = result
            progress_bar.progress(1.0)
            status_text.success(f"Screener complete! {len(result.picks)} picks ranked.")
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e) or repr(e)
            st.error(f"Screener failed ({err_type}): {err_msg}")
            with st.expander("Full error traceback"):
                st.code(traceback.format_exc())
            return

    result = st.session_state.get("screener_result")
    if result is None:
        st.info(
            "Enter a watchlist of tickers above and click **Run Screener** "
            "to rank stocks by trading opportunity."
        )
        return

    st.markdown("---")
    render_screener_results(result)

    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    exporter = ReportExporter()
    with col_dl1:
        screener_data = exporter.screener_to_dict(result)
        st.download_button(
            "Download JSON Report",
            data=json.dumps(screener_data, indent=2, default=str),
            file_name="screener_report.json",
            mime="application/json",
        )
    with col_dl2:
        screener_html = exporter._render_screener_html(exporter.screener_to_dict(result))
        st.download_button(
            "Download HTML Report",
            data=screener_html,
            file_name="screener_report.html",
            mime="text/html",
        )


if __name__ == "__main__":
    main()
