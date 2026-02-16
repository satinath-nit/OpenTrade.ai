from unittest.mock import MagicMock, patch

from opentrade_ai.data.google_news import GoogleNewsProvider
from opentrade_ai.data.google_trends import GoogleTrendsProvider
from opentrade_ai.data.sec_edgar import SECEdgarProvider


class TestGoogleNewsProvider:
    def setup_method(self):
        self.provider = GoogleNewsProvider()

    @patch("opentrade_ai.data.google_news.GNews")
    def test_get_news_returns_list(self, mock_gnews_cls):
        mock_client = MagicMock()
        mock_client.get_news.return_value = [
            {
                "title": "Apple stock surges on earnings beat",
                "description": "Apple reported Q4 earnings above expectations.",
                "published date": "2026-01-15 10:00:00",
                "publisher": {"title": "Reuters"},
                "url": "https://example.com/article1",
            },
            {
                "title": "Tech sector rally continues",
                "description": "Major tech stocks rise.",
                "published date": "2026-01-14 09:00:00",
                "publisher": {"title": "Bloomberg"},
                "url": "https://example.com/article2",
            },
        ]
        mock_gnews_cls.return_value = mock_client

        results = self.provider.get_news("AAPL")
        assert len(results) == 2
        assert results[0]["title"] == "Apple stock surges on earnings beat"
        assert results[0]["publisher"] == "Reuters"
        assert results[0]["url"] == "https://example.com/article1"

    @patch("opentrade_ai.data.google_news.GNews")
    def test_get_news_empty(self, mock_gnews_cls):
        mock_client = MagicMock()
        mock_client.get_news.return_value = []
        mock_gnews_cls.return_value = mock_client

        results = self.provider.get_news("INVALID")
        assert results == []

    @patch("opentrade_ai.data.google_news.GNews")
    def test_get_news_handles_exception(self, mock_gnews_cls):
        mock_client = MagicMock()
        mock_client.get_news.side_effect = Exception("Network error")
        mock_gnews_cls.return_value = mock_client

        results = self.provider.get_news("AAPL")
        assert results == []

    @patch("opentrade_ai.data.google_news.GNews")
    def test_get_news_missing_fields(self, mock_gnews_cls):
        mock_client = MagicMock()
        mock_client.get_news.return_value = [
            {"title": "Headline only"},
        ]
        mock_gnews_cls.return_value = mock_client

        results = self.provider.get_news("AAPL")
        assert len(results) == 1
        assert results[0]["title"] == "Headline only"
        assert results[0]["publisher"] == "Unknown"
        assert results[0]["description"] == ""

    @patch("opentrade_ai.data.google_news.GNews")
    def test_get_news_with_company_name(self, mock_gnews_cls):
        mock_client = MagicMock()
        mock_client.get_news.return_value = []
        mock_gnews_cls.return_value = mock_client

        self.provider.get_news("AAPL", company_name="Apple Inc.")
        call_args = mock_client.get_news.call_args[0][0]
        assert "Apple" in call_args or "AAPL" in call_args


class TestSECEdgarProvider:
    def setup_method(self):
        self.provider = SECEdgarProvider()

    @patch("opentrade_ai.data.sec_edgar.requests.get")
    def test_get_filings_returns_list(self, mock_get):
        self.provider._cik_cache["AAPL"] = "0000320193"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "accessionNumber": ["0001-23-456789", "0001-23-456790"],
                    "filingDate": ["2026-01-10", "2025-12-15"],
                    "primaryDocument": ["doc1.htm", "doc2.htm"],
                    "primaryDocDescription": ["10-K Annual Report", "10-Q Quarterly"],
                    "form": ["10-K", "10-Q"],
                }
            }
        }
        mock_get.return_value = mock_response

        results = self.provider.get_filings("AAPL")
        assert len(results) == 2
        assert results[0]["form"] == "10-K"
        assert results[0]["filing_date"] == "2026-01-10"
        assert results[0]["description"] == "10-K Annual Report"

    @patch("opentrade_ai.data.sec_edgar.requests.get")
    def test_get_filings_empty(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "accessionNumber": [],
                    "filingDate": [],
                    "primaryDocument": [],
                    "primaryDocDescription": [],
                    "form": [],
                }
            }
        }
        mock_get.return_value = mock_response

        results = self.provider.get_filings("INVALID")
        assert results == []

    @patch("opentrade_ai.data.sec_edgar.requests.get")
    def test_get_filings_handles_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not Found")
        mock_get.return_value = mock_response

        results = self.provider.get_filings("AAPL")
        assert results == []

    @patch("opentrade_ai.data.sec_edgar.requests.get")
    def test_get_filings_handles_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        results = self.provider.get_filings("AAPL")
        assert results == []

    @patch("opentrade_ai.data.sec_edgar.requests.get")
    def test_get_filings_filters_by_form_type(self, mock_get):
        self.provider._cik_cache["AAPL"] = "0000320193"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "accessionNumber": ["001", "002", "003"],
                    "filingDate": ["2026-01-10", "2025-12-15", "2025-11-01"],
                    "primaryDocument": ["d1.htm", "d2.htm", "d3.htm"],
                    "primaryDocDescription": ["10-K", "8-K", "10-Q"],
                    "form": ["10-K", "8-K", "10-Q"],
                }
            }
        }
        mock_get.return_value = mock_response

        results = self.provider.get_filings("AAPL", form_types=["10-K", "10-Q"])
        assert len(results) == 2
        forms = [r["form"] for r in results]
        assert "10-K" in forms
        assert "10-Q" in forms
        assert "8-K" not in forms

    def test_cik_lookup_returns_cached(self):
        self.provider._cik_cache["AAPL"] = "0000320193"
        cik = self.provider._get_cik("AAPL")
        assert cik == "0000320193"


class TestGoogleTrendsProvider:
    def setup_method(self):
        self.provider = GoogleTrendsProvider()

    @patch("opentrade_ai.data.google_trends.TrendReq")
    def test_get_interest_returns_dict(self, mock_trendreq_cls):
        mock_client = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(
            mean=MagicMock(return_value=65.0),
            iloc=MagicMock(__getitem__=MagicMock(return_value=72.0)),
            tolist=MagicMock(return_value=[60, 65, 70, 72]),
        ))
        mock_df.columns = ["AAPL stock"]
        mock_client.interest_over_time.return_value = mock_df
        mock_trendreq_cls.return_value = mock_client

        result = self.provider.get_interest("AAPL")
        assert "average_interest" in result
        assert "current_interest" in result
        assert "keyword" in result

    @patch("opentrade_ai.data.google_trends.TrendReq")
    def test_get_interest_empty(self, mock_trendreq_cls):
        mock_client = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = True
        mock_client.interest_over_time.return_value = mock_df
        mock_trendreq_cls.return_value = mock_client

        result = self.provider.get_interest("INVALIDTICKER")
        assert result["average_interest"] == 0
        assert result["trend"] == "no_data"

    @patch("opentrade_ai.data.google_trends.TrendReq")
    def test_get_interest_handles_exception(self, mock_trendreq_cls):
        mock_client = MagicMock()
        mock_client.build_payload.side_effect = Exception("Rate limited")
        mock_trendreq_cls.return_value = mock_client

        result = self.provider.get_interest("AAPL")
        assert result["average_interest"] == 0
        assert result["trend"] == "error"

    @patch("opentrade_ai.data.google_trends.TrendReq")
    def test_get_interest_with_company_name(self, mock_trendreq_cls):
        mock_client = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = True
        mock_client.interest_over_time.return_value = mock_df
        mock_trendreq_cls.return_value = mock_client

        self.provider.get_interest("AAPL", company_name="Apple Inc.")
        call_args = mock_client.build_payload.call_args
        keywords = call_args[0][0]
        assert any("Apple" in kw for kw in keywords)

    @patch("opentrade_ai.data.google_trends.TrendReq")
    def test_trend_direction_rising(self, mock_trendreq_cls):
        mock_client = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        series_mock = MagicMock()
        series_mock.mean.return_value = 50.0
        series_mock.iloc.__getitem__ = MagicMock(side_effect=[30.0, 70.0])
        series_mock.tolist.return_value = [30, 40, 50, 60, 70]
        mock_df.__getitem__ = MagicMock(return_value=series_mock)
        mock_df.columns = ["AAPL stock"]
        mock_client.interest_over_time.return_value = mock_df
        mock_trendreq_cls.return_value = mock_client

        result = self.provider.get_interest("AAPL")
        assert result["trend"] in ("rising", "stable", "declining")
