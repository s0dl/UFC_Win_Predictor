from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import requests


SCRAPING_DIR = Path(__file__).resolve().parent
if str(SCRAPING_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPING_DIR))

from scrape_bestfightodds_moneylines import (  # noqa: E402
    latest_ufc_event_urls,
    scrape_event_moneylines,
)


@dataclass
class NextEventCard:
    event: str
    date: str
    source_url: str
    scraped_at: str
    fights: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "date": self.date,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at,
            "fights": self.fights,
        }


def parse_event_date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def normalize_fight(record: dict[str, Any], source_order: int, importance_order: int) -> dict[str, Any]:
    return {
        "source_order": source_order,
        "importance_order": importance_order,
        "fighter1": record.get("fighter1"),
        "fighter2": record.get("fighter2"),
        "fighter1_open_odds": record.get("f1_open"),
        "fighter2_open_odds": record.get("f2_open"),
        "fighter1_current_odds": record.get("f1_close") or record.get("f1_odds"),
        "fighter2_current_odds": record.get("f2_close") or record.get("f2_odds"),
        "fighter1_open_time": record.get("f1_open_ts"),
        "fighter2_open_time": record.get("f2_open_ts"),
        "fighter1_current_time": record.get("f1_close_ts"),
        "fighter2_current_time": record.get("f2_close_ts"),
        "matchup_id": record.get("matchup_id"),
    }


def scrape_next_ufc_event(
    max_event_pages: int = 18,
    delay: float = 0.35,
    main_event_position: str = "top",
) -> NextEventCard:
    session = requests.Session()
    urls = latest_ufc_event_urls(session)
    if not urls:
        raise RuntimeError("No upcoming UFC event URLs found on BestFightOdds latest odds page.")

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    fallback_card: NextEventCard | None = None
    future_cards: list[tuple[pd.Timestamp, NextEventCard]] = []

    for url in urls[:max_event_pages]:
        records = scrape_event_moneylines(url, session)
        if not records:
            time.sleep(delay)
            continue

        event_date = parse_event_date(records[0].get("date"))
        indexed_records = list(enumerate(records, start=1))
        ordered_records = list(reversed(indexed_records)) if main_event_position == "bottom" else indexed_records
        fights = [
            normalize_fight(
                record,
                source_index,
                importance_index,
            )
            for importance_index, (source_index, record) in enumerate(ordered_records, start=1)
        ]
        card = NextEventCard(
            event=str(records[0].get("event") or "Upcoming UFC event"),
            date=str(records[0].get("date") or ""),
            source_url=url,
            scraped_at=datetime.now(timezone.utc).isoformat(),
            fights=fights,
        )

        if event_date is not None and event_date >= today:
            future_cards.append((event_date, card))
        elif event_date is None and fallback_card is None:
            fallback_card = card
        elif fallback_card is None:
            fallback_card = card

        time.sleep(delay)

    if future_cards:
        return sorted(future_cards, key=lambda item: item[0])[0][1]
    if fallback_card is not None:
        return fallback_card
    raise RuntimeError("Could not scrape fight odds for the next UFC event.")
