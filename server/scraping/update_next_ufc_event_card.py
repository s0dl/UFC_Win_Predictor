"""Scrape and cache the next UFC event card with current BestFightOdds lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scrape_next_ufc_event import scrape_next_ufc_event
from scraping_common import DATA_DIR, write_csv_atomic


DEFAULT_OUTPUT = DATA_DIR / "ufc_next_event_card.csv"


def build_next_event_card(max_event_pages: int, delay: float, main_event_position: str) -> pd.DataFrame:
    card = scrape_next_ufc_event(
        max_event_pages=max_event_pages,
        delay=delay,
        main_event_position=main_event_position,
    ).as_dict()
    rows = []
    for fight in card["fights"]:
        rows.append(
            {
                "event": card["event"],
                "date": card["date"],
                "source_url": card["source_url"],
                "scraped_at": card["scraped_at"],
                **fight,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update cached next UFC event card odds.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-event-pages", type=int, default=18)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument(
        "--main-event-position",
        choices=["top", "bottom"],
        default="top",
        help="Where BestFightOdds lists the main event on the event page.",
    )
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = build_next_event_card(args.max_event_pages, args.delay, args.main_event_position)
    print(f"Built {args.output} with shape {df.shape}")
    if not df.empty:
        first = df.iloc[0]
        print(f"Main event: {first['fighter1']} vs {first['fighter2']}")
    if not args.dry_run:
        write_csv_atomic(df, args.output, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
