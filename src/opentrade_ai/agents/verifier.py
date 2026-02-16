import json

from opentrade_ai.agents.base import AgentRole, AnalysisResult, BaseAgent
from opentrade_ai.llm.provider import LLMProvider


class VerifierAgent(BaseAgent):
    role = AgentRole.VERIFIER
    system_prompt = (
        "You are the Chief Quality Officer at a top-tier quantitative trading firm. "
        "You are the final gate before any trade recommendation reaches the portfolio "
        "manager. Your job is to catch errors, inconsistencies, and blind spots.\n\n"
        "VERIFICATION CHECKLIST:\n"
        "1. Signal-Reasoning Consistency: Does each analyst's signal match their "
        "stated reasoning? (e.g., analyst says 'declining revenue' but signals BUY)\n"
        "2. Cross-Analyst Contradictions: Are there conflicting signals that the "
        "trader failed to reconcile? Flag unresolved conflicts.\n"
        "3. Data Support: Are key claims backed by actual numbers from the data? "
        "Flag unsupported assertions (e.g., 'strong growth' without citing figures).\n"
        "4. Missing Risk Factors: Did the risk manager miss obvious risks? "
        "Check for: sector concentration, earnings date proximity, macro events.\n"
        "5. Confidence Calibration: Is the final confidence level justified? "
        "High confidence (>80%) should require strong consensus across analysts.\n"
        "6. Bias Detection: Is the final decision overly influenced by one analyst "
        "while ignoring valid counterarguments from others?\n\n"
        "EXAMPLE OUTPUT:\n"
        '{"verdict": "FLAGGED", "confidence_adjustment": -15, '
        '"issues": ["Fundamental analyst signals BUY citing 20% revenue growth, '
        "but news analyst reports CFO departure (material event not addressed)\", "
        '"\"Trader confidence at 85% despite 2/4 analysts being neutral - '
        'overconfident\"], "summary": "Analysis pipeline has two significant '
        "gaps: (1) material news event (CFO departure) not factored into "
        "fundamental analysis, (2) confidence level not justified by analyst "
        "consensus. Recommend reducing confidence by 15 points and re-evaluating "
        'after incorporating management change implications."}\n\n'
        "You MUST respond with a JSON object:\n"
        '{"verdict": "APPROVED|FLAGGED|REJECTED", '
        '"confidence_adjustment": <int from -30 to 0>, '
        '"issues": [<list of specific issues>], '
        '"summary": "<your detailed assessment>"}'
    )

    def __init__(self, llm: LLMProvider):
        super().__init__(llm)

    def analyze(self, ticker: str, context: dict) -> AnalysisResult:
        return self.verify(
            ticker=ticker,
            analyst_reports=context.get("analyst_reports", []),
            trader_summary=context.get("trader_summary", ""),
            risk_assessment=context.get("risk_assessment", ""),
        )

    def verify(
        self,
        ticker: str,
        analyst_reports: list[AnalysisResult],
        trader_summary: str,
        risk_assessment: str,
    ) -> AnalysisResult:
        prompt = self._build_verify_prompt(
            ticker, analyst_reports, trader_summary, risk_assessment
        )
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_verify_response(ticker, response)

    def _build_verify_prompt(
        self,
        ticker: str,
        analyst_reports: list[AnalysisResult],
        trader_summary: str,
        risk_assessment: str,
    ) -> str:
        parts = [
            f"Verify the analysis pipeline output for {ticker}.",
            "\n--- ANALYST REPORTS ---",
        ]

        if analyst_reports:
            for report in analyst_reports:
                role_val = report.agent_role
                role = role_val.value if hasattr(role_val, "value") else str(role_val)
                parts.append(
                    f"\n[{role}] Signal: {report.signal} | "
                    f"Confidence: {report.confidence}%"
                )
                parts.append(f"Summary: {report.summary[:400]}")
        else:
            parts.append("No analyst reports provided.")

        parts.append("\n--- TRADER DECISION ---")
        parts.append(trader_summary[:500] if trader_summary else "No trader summary.")

        parts.append("\n--- RISK ASSESSMENT ---")
        parts.append(risk_assessment[:500] if risk_assessment else "No risk assessment.")

        parts.append(
            "\nReview for consistency, contradictions, unsupported claims, "
            "missing risks, and bias. Respond with JSON: "
            '{"verdict": "APPROVED|FLAGGED|REJECTED", '
            '"confidence_adjustment": <int -30 to 0>, '
            '"issues": [<list>], "summary": "<assessment>"}'
        )
        return "\n".join(parts)

    def _build_prompt(self, ticker: str, context: dict) -> str:
        return self._build_verify_prompt(
            ticker,
            context.get("analyst_reports", []),
            context.get("trader_summary", ""),
            context.get("risk_assessment", ""),
        )

    def _parse_verify_response(self, ticker: str, response: str) -> AnalysisResult:
        parsed = None
        text = response.strip()

        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        elif "```json" in text:
            try:
                block = text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(block)
            except (IndexError, json.JSONDecodeError):
                parsed = None

        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).lower().strip()
            issues = parsed.get("issues", [])
            confidence_adj = parsed.get("confidence_adjustment", 0)
            summary = str(parsed.get("summary", response))

            signal = self._verdict_to_signal(verdict)

            return AnalysisResult(
                agent_role=self.role,
                ticker=ticker,
                summary=summary,
                signal=signal,
                confidence=max(0.0, 100.0 + float(confidence_adj)),
                details={
                    "verdict": verdict,
                    "issues": issues if isinstance(issues, list) else [],
                    "confidence_adjustment": confidence_adj,
                },
            )

        signal = self._verdict_from_text(response)
        return AnalysisResult(
            agent_role=self.role,
            ticker=ticker,
            summary=response,
            signal=signal,
            confidence=70.0,
            details={"verdict": signal, "issues": [], "confidence_adjustment": 0},
        )

    def _verdict_to_signal(self, verdict: str) -> str:
        if "approved" in verdict or "approve" in verdict:
            return "approved"
        if "rejected" in verdict or "reject" in verdict:
            return "rejected"
        if "flagged" in verdict or "flag" in verdict:
            return "flagged"
        return "flagged"

    def _verdict_from_text(self, response: str) -> str:
        lower = response.lower()
        if "approved" in lower or "approve" in lower:
            return "approved"
        if "rejected" in lower or "reject" in lower:
            return "rejected"
        return "flagged"
