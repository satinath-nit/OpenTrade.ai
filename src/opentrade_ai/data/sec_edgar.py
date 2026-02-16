import logging

import requests

logger = logging.getLogger(__name__)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DEFAULT_FORM_TYPES = ("10-K", "10-Q", "8-K")
_HEADERS = {"User-Agent": "OpenTrade.ai research@example.com"}


class SECEdgarProvider:
    def __init__(self, max_filings: int = 10):
        self._max_filings = max_filings
        self._cik_cache: dict[str, str] = {}

    def _get_cik(self, ticker: str) -> str | None:
        upper = ticker.upper()
        if upper in self._cik_cache:
            return self._cik_cache[upper]
        try:
            resp = requests.get(_COMPANY_TICKERS_URL, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                t = str(entry.get("ticker", "")).upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                self._cik_cache[t] = cik
                if t == upper:
                    return cik
        except Exception:
            logger.debug("CIK lookup failed for %s", ticker, exc_info=True)
        return None

    def get_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
    ) -> list[dict]:
        allowed = set(form_types) if form_types else set(_DEFAULT_FORM_TYPES)
        try:
            cik = self._get_cik(ticker)
            if not cik:
                return []
            url = _SUBMISSIONS_URL.format(cik=cik)
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            docs = recent.get("primaryDocument", [])
            descs = recent.get("primaryDocDescription", [])

            results = []
            for i in range(len(forms)):
                if forms[i] not in allowed:
                    continue
                acc = accessions[i].replace("-", "") if i < len(accessions) else ""
                doc = docs[i] if i < len(docs) else ""
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{acc}/{doc}"
                ) if acc and doc else ""
                results.append({
                    "form": forms[i],
                    "filing_date": dates[i] if i < len(dates) else "",
                    "description": descs[i] if i < len(descs) else "",
                    "accession_number": accessions[i] if i < len(accessions) else "",
                    "url": filing_url,
                    "source": "sec_edgar",
                })
                if len(results) >= self._max_filings:
                    break
            return results
        except Exception:
            logger.debug("SEC EDGAR fetch failed for %s", ticker, exc_info=True)
            return []
