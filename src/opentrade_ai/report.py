from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from html import escape


class ReportExporter:
    def decision_to_dict(self, decision) -> dict:
        reports = []
        for r in decision.analyst_reports:
            role_val = r.agent_role
            role = role_val.value if hasattr(role_val, "value") else str(role_val)
            reports.append({
                "agent": role,
                "signal": r.signal,
                "confidence": r.confidence,
                "summary": r.summary,
            })

        return {
            "run_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now().isoformat(),
            "ticker": decision.ticker,
            "final_signal": decision.final_signal,
            "confidence": decision.confidence,
            "trader_summary": decision.trader_summary,
            "risk_assessment": decision.risk_assessment,
            "analyst_reports": reports,
            "debate_history": decision.debate_history,
            "stock_info": decision.stock_info,
            "indicators": self._safe_indicators(decision.indicators),
        }

    def screener_to_dict(self, result) -> dict:
        return result.to_dict()

    def save_json(self, decision, output_dir: str = "reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        data = self.decision_to_dict(decision)
        filename = f"{decision.ticker}_{data['run_id']}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_html(self, decision, output_dir: str = "reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        data = self.decision_to_dict(decision)
        html = self._render_decision_html(data)
        filename = f"{decision.ticker}_{data['run_id']}.html"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(html)
        return path

    def save_screener_json(self, result, output_dir: str = "reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        data = self.screener_to_dict(result)
        filename = f"screener_{data['run_id']}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_screener_html(self, result, output_dir: str = "reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        data = self.screener_to_dict(result)
        html = self._render_screener_html(data)
        filename = f"screener_{data['run_id']}.html"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(html)
        return path

    def _safe_indicators(self, indicators: dict) -> dict:
        safe = {}
        for k, v in indicators.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
        return safe

    def _signal_color(self, signal: str) -> str:
        colors = {
            "strong_buy": "#00C853",
            "buy": "#4CAF50",
            "hold": "#FFC107",
            "sell": "#FF5722",
            "strong_sell": "#D50000",
        }
        return colors.get(signal, "#9E9E9E")

    def _render_decision_html(self, data: dict) -> str:
        signal = data["final_signal"]
        color = self._signal_color(signal)
        ticker = escape(data["ticker"])
        stock_name = escape(str(data.get("stock_info", {}).get("name", ticker)))

        reports_html = ""
        for r in data.get("analyst_reports", []):
            rc = self._signal_color(r["signal"])
            reports_html += (
                f'<div class="card">'
                f'<h3>{escape(r["agent"].replace("_", " ").title())}</h3>'
                f'<span class="badge" style="background:{rc}">'
                f'{escape(r["signal"].replace("_", " ").upper())}</span> '
                f'<span>Confidence: {r["confidence"]:.0f}%</span>'
                f'<p>{escape(r["summary"][:500])}</p></div>'
            )

        debate_html = ""
        for i, entry in enumerate(data.get("debate_history", [])):
            bull_text = escape(str(entry.get("bull", ""))[:400])
            bear_text = escape(str(entry.get("bear", ""))[:400])
            debate_html += (
                f'<div class="debate-round">'
                f'<h3>Round {i+1}</h3>'
                f'<div class="bull"><strong>Bull:</strong> {bull_text}</div>'
                f'<div class="bear"><strong>Bear:</strong> {bear_text}</div>'
                f'</div>'
            )

        indicators_html = ""
        for k, v in data.get("indicators", {}).items():
            indicators_html += f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Report - {ticker}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
.header {{ text-align: center; padding: 30px; border-radius: 12px;
           background: linear-gradient(135deg, {color}22, {color}44); border: 2px solid {color}; }}
.header h1 {{ color: {color}; margin: 0; font-size: 2.5em; }}
.header .confidence {{ color: #ccc; font-size: 1.2em; }}
.card {{ background: #16213e; padding: 16px; border-radius: 8px; margin: 10px 0; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white;
          font-weight: bold; font-size: 0.85em; }}
.debate-round {{ background: #16213e; padding: 16px; border-radius: 8px; margin: 10px 0; }}
.bull {{ border-left: 4px solid #4CAF50; padding-left: 12px; margin: 8px 0; }}
.bear {{ border-left: 4px solid #EF5350; padding-left: 12px; margin: 8px 0; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #333; }}
h2 {{ color: #64b5f6; border-bottom: 2px solid #64b5f6; padding-bottom: 8px; }}
.meta {{ color: #888; font-size: 0.85em; text-align: center; margin-top: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>{escape(signal.replace('_', ' ').upper())}</h1>
  <div class="confidence">Confidence: {data['confidence']:.1f}%</div>
  <p>{stock_name} ({ticker})</p>
</div>

<h2>Analyst Reports</h2>
{reports_html}

<h2>Trader Summary</h2>
<div class="card"><p>{escape(data.get('trader_summary', 'N/A')[:800])}</p></div>

<h2>Bull vs Bear Debate</h2>
{debate_html if debate_html else '<p>No debate history.</p>'}

<h2>Risk Assessment</h2>
<div class="card"><p>{escape(data.get('risk_assessment', 'N/A')[:800])}</p></div>

<h2>Technical Indicators</h2>
<table><tr><th>Indicator</th><th>Value</th></tr>{indicators_html}</table>

<div class="meta">
  Generated: {escape(data['timestamp'])} | Run ID: {escape(data['run_id'])} |
  OpenTrade.ai (not financial advice)
</div>
</body></html>"""

    def _render_screener_html(self, data: dict) -> str:
        rows_html = ""
        for p in data.get("picks", []):
            color = self._signal_color(p["signal"])
            risks = ", ".join(p.get("key_risks", []))
            rows_html += (
                f"<tr>"
                f'<td>{p["rank"]}</td>'
                f'<td><strong>{escape(p["ticker"])}</strong></td>'
                f'<td><span class="badge" style="background:{color}">'
                f'{escape(p["signal"].replace("_", " ").upper())}</span></td>'
                f'<td>{p["confidence"]:.0f}%</td>'
                f'<td>{escape(p.get("rationale", "")[:200])}</td>'
                f'<td>{p.get("position_size_pct", 0):.1f}%</td>'
                f'<td>{escape(p.get("time_horizon", ""))}</td>'
                f'<td>{escape(risks)}</td>'
                f"</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Screener Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #64b5f6; text-align: center; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th {{ background: #16213e; padding: 12px 8px; text-align: left; color: #64b5f6; }}
td {{ padding: 10px 8px; border-bottom: 1px solid #333; }}
tr:hover {{ background: #16213e44; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; color: white;
          font-weight: bold; font-size: 0.8em; }}
.meta {{ color: #888; font-size: 0.85em; text-align: center; margin-top: 20px; }}
</style>
</head>
<body>
<h1>Stock Screener Report</h1>
<p style="text-align:center; color:#aaa;">
  Watchlist: {escape(', '.join(data.get('watchlist', [])))} |
  {len(data.get('picks', []))} picks ranked
</p>
<table>
<tr><th>#</th><th>Ticker</th><th>Signal</th><th>Confidence</th>
<th>Rationale</th><th>Position</th><th>Horizon</th><th>Risks</th></tr>
{rows_html}
</table>
<div class="meta">
  Generated: {escape(data.get('timestamp', ''))} | Run ID: {escape(data.get('run_id', ''))} |
  OpenTrade.ai (not financial advice)
</div>
</body></html>"""
