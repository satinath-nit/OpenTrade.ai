import sys
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from opentrade_ai.config import AppConfig
from opentrade_ai.graph.trading_graph import OpenTradeGraph
from opentrade_ai.report import ReportExporter
from opentrade_ai.screener import OpenTradeScreener, parse_watchlist_input

console = Console()


def print_banner():
    console.print(
        Panel(
            "[bold cyan]OpenTrade.ai[/bold cyan]\n"
            "[dim]Multi-Agent LLM Stock Trading Advisor[/dim]",
            border_style="cyan",
        )
    )


def print_decision(decision):
    signal_colors = {
        "strong_buy": "bold green",
        "buy": "green",
        "hold": "yellow",
        "sell": "red",
        "strong_sell": "bold red",
    }
    color = signal_colors.get(decision.final_signal, "white")

    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(
        f"[bold]Final Decision for {decision.ticker}[/bold]: "
        f"[{color}]{decision.final_signal.upper().replace('_', ' ')}[/{color}]"
    )
    console.print(f"Confidence: {decision.confidence:.1f}%")
    console.print(f"[bold]{'=' * 60}[/bold]")

    if decision.stock_info:
        info = decision.stock_info
        table = Table(title="Stock Info")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        if info.get("name"):
            table.add_row("Name", str(info["name"]))
        if info.get("sector"):
            table.add_row("Sector", str(info["sector"]))
        if info.get("current_price"):
            table.add_row("Price", f"${info['current_price']:.2f}")
        if info.get("market_cap"):
            table.add_row("Market Cap", f"${info['market_cap']:,.0f}")
        if info.get("pe_ratio"):
            table.add_row("P/E Ratio", f"{info['pe_ratio']:.2f}")
        console.print(table)

    if decision.indicators:
        table = Table(title="Technical Indicators")
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", style="white")
        ind = decision.indicators
        if ind.get("rsi") is not None:
            table.add_row("RSI", f"{ind['rsi']:.2f}")
        if ind.get("macd") is not None:
            table.add_row("MACD", f"{ind['macd']:.4f}")
        if ind.get("sma_20") is not None:
            table.add_row("SMA 20", f"${ind['sma_20']:.2f}")
        if ind.get("sma_50") is not None:
            table.add_row("SMA 50", f"${ind['sma_50']:.2f}")
        if ind.get("volume_trend") is not None:
            table.add_row("Volume Trend", f"{ind['volume_trend']:.2f}x")
        console.print(table)

    if decision.analyst_reports:
        table = Table(title="Analyst Reports")
        table.add_column("Agent", style="cyan")
        table.add_column("Signal", style="white")
        table.add_column("Confidence", style="white")
        for report in decision.analyst_reports:
            role_val = report.agent_role
            role = role_val.value if hasattr(role_val, "value") else str(role_val)
            table.add_row(role, report.signal, f"{report.confidence:.0f}%")
        console.print(table)

    if decision.trader_summary:
        console.print(
            Panel(decision.trader_summary[:800], title="Trader Analysis", border_style="green")
        )

    if decision.risk_assessment:
        console.print(
            Panel(decision.risk_assessment[:800], title="Risk Assessment", border_style="red")
        )

    if decision.verification_summary:
        console.print(
            Panel(
                decision.verification_summary[:800],
                title="Verification",
                border_style="blue",
            )
        )
        if decision.verification_issues:
            console.print("[bold blue]Issues found:[/bold blue]")
            for issue in decision.verification_issues:
                console.print(f"  [dim]- {issue}[/dim]")


def print_screener_result(result):
    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning: {err}[/yellow]")

    if not result.picks:
        console.print("[red]No picks available.[/red]")
        return

    table = Table(title="Screener Results - Top Picks")
    table.add_column("#", style="bold")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Signal", style="white")
    table.add_column("Confidence", style="white")
    table.add_column("Rationale", style="dim")
    table.add_column("Position %", style="white")
    table.add_column("Horizon", style="white")
    table.add_column("Key Risks", style="dim")

    signal_colors = {
        "strong_buy": "bold green",
        "buy": "green",
        "hold": "yellow",
        "sell": "red",
        "strong_sell": "bold red",
    }

    for pick in result.picks:
        color = signal_colors.get(pick.signal, "white")
        risks = ", ".join(pick.key_risks[:3]) if pick.key_risks else "N/A"
        table.add_row(
            str(pick.rank),
            pick.ticker,
            f"[{color}]{pick.signal.upper().replace('_', ' ')}[/{color}]",
            f"{pick.confidence:.0f}%",
            pick.rationale[:80] + ("..." if len(pick.rationale) > 80 else ""),
            f"{pick.position_size_pct:.1f}%",
            pick.time_horizon or "N/A",
            risks,
        )

    console.print(table)


@click.command()
@click.option("--ticker", "-t", default=None, help="Stock ticker (e.g., AAPL)")
@click.option("--date", "-d", default=None, help="Analysis date (YYYY-MM-DD)")
@click.option("--provider", "-p", default=None, help="LLM provider (ollama/lmstudio/openai)")
@click.option("--model", "-m", default=None, help="LLM model name")
@click.option(
    "--risk", "-r", default="moderate",
    help="Risk tolerance (conservative/moderate/aggressive)",
)
@click.option(
    "--screener", "-s", default=None,
    help="Screener mode: comma-separated tickers (e.g., AAPL,MSFT,GOOGL)",
)
@click.option("--top-n", default=10, help="Number of top picks in screener mode")
@click.option(
    "--export", "-e", is_flag=True, default=False,
    help="Export report (JSON + HTML) to reports/ directory",
)
def main(ticker, date, provider, model, risk, screener, top_n, export):
    print_banner()

    config = AppConfig.from_env()
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    config.trading.risk_tolerance = risk

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
        console.print(f"[dim]Using current date: {date}[/dim]")

    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"[red]Config error: {err}[/red]")
        sys.exit(1)

    if screener:
        _run_screener(config, screener, date, top_n, export)
    else:
        if not ticker:
            ticker = console.input("[cyan]Enter stock ticker:[/cyan] ").strip().upper()
            if not ticker:
                console.print("[red]No ticker provided. Exiting.[/red]")
                sys.exit(1)
        _run_single_analysis(config, ticker, date, export)


def _run_single_analysis(config, ticker, date, export):
    console.print(f"\n[bold]Analyzing {ticker}...[/bold]")
    console.print(
        f"[dim]Provider: {config.llm.provider} | Model: {config.llm.model} "
        f"| Risk: {config.trading.risk_tolerance}[/dim]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        current_task = [None]

        def on_step(step):
            if step.status == "pending":
                current_task[0] = progress.add_task(f"{step.step_name}...", total=None)
            elif step.status == "completed" and current_task[0] is not None:
                desc = f"[green]{step.step_name} - Done[/green]"
                progress.update(current_task[0], description=desc)
                progress.stop_task(current_task[0])
            elif step.status == "error" and current_task[0] is not None:
                progress.update(
                    current_task[0],
                    description=f"[red]{step.step_name} - Error: {step.error}[/red]",
                )
                progress.stop_task(current_task[0])

        graph = OpenTradeGraph(config, on_step=on_step)

        try:
            decision = graph.propagate(ticker, date)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

    print_decision(decision)

    if export:
        exporter = ReportExporter()
        json_path = exporter.save_json(decision)
        html_path = exporter.save_html(decision)
        console.print("\n[green]Report exported:[/green]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  HTML: {html_path}")


def _run_screener(config, screener_input, date, top_n, export):
    tickers = parse_watchlist_input(screener_input)
    if not tickers:
        console.print("[red]No valid tickers provided.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Running Screener on {len(tickers)} tickers...[/bold]")
    console.print(f"[dim]Tickers: {', '.join(tickers)}[/dim]")
    console.print(
        f"[dim]Provider: {config.llm.provider} | Model: {config.llm.model} "
        f"| Risk: {config.trading.risk_tolerance}[/dim]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Screening...", total=None)

        def on_progress(msg):
            try:
                progress.update(task, description=msg)
            except Exception:
                pass

        screener_obj = OpenTradeScreener(config, on_progress=on_progress)

        try:
            result = screener_obj.screen(tickers, date, top_n)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

        progress.stop_task(task)

    print_screener_result(result)

    if export:
        exporter = ReportExporter()
        json_path = exporter.save_screener_json(result)
        html_path = exporter.save_screener_html(result)
        console.print("\n[green]Screener report exported:[/green]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  HTML: {html_path}")


if __name__ == "__main__":
    main()
