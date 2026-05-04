"""
Update server/models/data/ufc-fighters-statistics.csv from UFCStats.

By default this script refreshes every discovered fighter profile because records
and rate stats can change for existing fighters. Use --only-new for a faster
append-only update.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import scrape_ufc_stats_library as lib
from scraping_common import CACHE_DIR, DATA_DIR, write_csv_atomic


DEFAULT_OUTPUT = DATA_DIR / "ufc-fighters-statistics.csv"
DEFAULT_FIGHTER_DETAILS = CACHE_DIR / "ufc_fighter_details.csv"
DEFAULT_DELAY_SECONDS = 0.15


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
    )
    return session


def get_soup(session: requests.Session, url: str, timeout: int) -> BeautifulSoup:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def discover_fighters(
    session: requests.Session,
    timeout: int,
    delay_seconds: float,
) -> pd.DataFrame:
    fighter_details = []

    for url in tqdm(lib.generate_alphabetical_urls(), desc="Discovering fighters"):
        soup = get_soup(session, url, timeout)
        fighter_details.append(
            lib.parse_fighter_details(soup, ["FIRST", "LAST", "NICKNAME", "URL"])
        )
        if delay_seconds:
            time.sleep(delay_seconds)

    if not fighter_details:
        return pd.DataFrame(columns=["FIRST", "LAST", "NICKNAME", "URL"])

    fighters = pd.concat(fighter_details, ignore_index=True)
    fighters = fighters.drop_duplicates(subset=["URL"], keep="first")
    fighters = fighters.sort_values(["LAST", "FIRST", "URL"], kind="stable")
    return fighters.reset_index(drop=True)


def existing_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    existing = pd.read_csv(csv_path, usecols=["name"])
    return set(existing["name"].dropna().astype(str))


def urls_for_only_new(fighters: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    names = existing_names(csv_path)

    if not names:
        return fighters

    full_names = (
        fighters["FIRST"].fillna("").astype(str).str.strip()
        + " "
        + fighters["LAST"].fillna("").astype(str).str.strip()
    ).str.strip()
    return fighters.loc[~full_names.isin(names)].reset_index(drop=True)


def scrape_fighter_statistics(
    session: requests.Session,
    fighter_urls: Iterable[str],
    timeout: int,
    delay_seconds: float,
) -> pd.DataFrame:
    rows = []

    for url in tqdm(list(fighter_urls), desc="Scraping fighter profiles"):
        soup = get_soup(session, url, timeout)
        rows.append(lib.parse_fighter_statistics(soup))
        if delay_seconds:
            time.sleep(delay_seconds)

    if not rows:
        return pd.DataFrame(columns=lib.FIGHTER_STATISTICS_COLUMN_NAMES)

    return pd.concat(rows, ignore_index=True)


def coerce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "wins",
        "losses",
        "draws",
        "height_cm",
        "weight_in_kg",
        "reach_in_cm",
        "significant_strikes_landed_per_minute",
        "significant_striking_accuracy",
        "significant_strikes_absorbed_per_minute",
        "significant_strike_defence",
        "average_takedowns_landed_per_15_minutes",
        "takedown_accuracy",
        "takedown_defense",
        "average_submissions_attempted_per_15_minutes",
    ]
    cleaned = df.copy()

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update ufc-fighters-statistics.csv from UFCStats fighter profiles."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV to update. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--fighter-details-output",
        type=Path,
        default=DEFAULT_FIGHTER_DETAILS,
        help=f"Discovered fighter URL cache. Default: {DEFAULT_FIGHTER_DETAILS}",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only scrape fighters whose full name is not already in the output CSV.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak copy of the previous output CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and scrape, but do not write output files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Delay between UFCStats requests. Default: {DEFAULT_DELAY_SECONDS}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = build_session()

    fighters = discover_fighters(session, args.timeout, args.delay)
    if fighters.empty:
        print("No fighter profiles were discovered.", file=sys.stderr)
        return 1

    if args.only_new:
        fighters_to_scrape = urls_for_only_new(fighters, args.output)
    else:
        fighters_to_scrape = fighters

    if fighters_to_scrape.empty:
        print("No new fighters to scrape.")
        return 0

    scraped = scrape_fighter_statistics(
        session,
        fighters_to_scrape["URL"],
        args.timeout,
        args.delay,
    )
    scraped = coerce_column_types(scraped)

    if args.only_new and args.output.exists():
        existing = pd.read_csv(args.output)
        updated = pd.concat([existing, scraped], ignore_index=True)
        updated = updated.drop_duplicates(subset=["name"], keep="last")
    else:
        updated = scraped

    updated = updated[lib.FIGHTER_STATISTICS_COLUMN_NAMES].reset_index(drop=True)

    print(
        f"Discovered {len(fighters)} fighters; scraped {len(scraped)}; "
        f"output rows: {len(updated)}."
    )

    if args.dry_run:
        print("Dry run selected; no files written.")
        return 0

    write_csv_atomic(fighters, args.fighter_details_output, backup=not args.no_backup)
    write_csv_atomic(updated, args.output, backup=not args.no_backup)
    print(f"Updated {args.output}")
    print(f"Updated {args.fighter_details_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
