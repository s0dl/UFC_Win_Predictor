from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPING_ROOT = REPO_ROOT / "server" / "scraping"
if str(SCRAPING_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRAPING_ROOT))

import scrape_bestfightodds_moneylines as bfo
import scrape_next_ufc_event as next_event


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def test_latest_ufc_event_urls_skips_bjj_links(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <a href="/events/ufc-bjj-8-4197">UFC BJJ 8</a>
        <a href="/events/ufc-319-4201">UFC 319</a>
      </body>
    </html>
    """

    monkeypatch.setattr(
        bfo,
        "get_with_retries",
        lambda session, url, retries=3, timeout=30: FakeResponse(html),
    )

    urls = bfo.latest_ufc_event_urls(object())

    assert urls == ["https://www.bestfightodds.com/events/ufc-319-4201"]


def test_scrape_next_ufc_event_skips_non_mma_cards(monkeypatch) -> None:
    bjj_url = "https://www.bestfightodds.com/events/ufc-bjj-8-4197"
    mma_url = "https://www.bestfightodds.com/events/ufc-319-4201"

    def fake_latest_ufc_event_urls(session):
        return [bjj_url, mma_url]

    def fake_scrape_event_moneylines(url, session):
        if url == bjj_url:
            return [
                {
                    "event": "UFC BJJ 8",
                    "date": "2026-05-22",
                    "fighter1": "Kevin Dantzler",
                    "fighter2": "Mikey Musumeci",
                    "f1_open": -500,
                    "f2_open": 300,
                    "f1_close": -550,
                    "f2_close": 350,
                    "f1_odds": -550,
                    "f2_odds": 350,
                    "matchup_id": "1",
                }
            ]

        return [
            {
                "event": "UFC 319",
                "date": "2026-05-30",
                "fighter1": "Main A",
                "fighter2": "Main B",
                "f1_open": -140,
                "f2_open": 120,
                "f1_close": -150,
                "f2_close": 130,
                "f1_odds": -150,
                "f2_odds": 130,
                "matchup_id": "2",
            }
        ]

    monkeypatch.setattr(next_event, "latest_ufc_event_urls", fake_latest_ufc_event_urls)
    monkeypatch.setattr(next_event, "scrape_event_moneylines", fake_scrape_event_moneylines)
    monkeypatch.setattr(next_event.time, "sleep", lambda *_args, **_kwargs: None)

    card = next_event.scrape_next_ufc_event(max_event_pages=2, delay=0.0)

    assert card.event == "UFC 319"
    assert card.source_url == mma_url
    assert card.fights[0]["fighter1"] == "Main A"
