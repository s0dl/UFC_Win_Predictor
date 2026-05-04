"""Update server/models/data/ufc_fight_results_with_odds.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

from combine_bestfightodds_with_results import build_results_with_odds
from scrape_bestfightodds_moneylines import read_event_urls, read_completed_source_urls, scrape_event_moneylines
from scraping_common import CACHE_DIR, DATA_DIR, write_csv_atomic
from update_ufc_fight_stats import DEFAULT_EVENTS_OUTPUT, DEFAULT_RESULTS_OUTPUT, update_ufcstats_fight_data


DEFAULT_ODDS_OUTPUT = CACHE_DIR / "bestfightodds_moneylines_full_open_close.csv"
DEFAULT_DISCOVERED_URLS_OUTPUT = CACHE_DIR / "bestfightodds_discovered_event_urls.txt"
DEFAULT_OUTPUT = DATA_DIR / "ufc_fight_results_with_odds.csv"


def ensure_bestfightodds_moneylines(
    odds_output: Path,
    events_path: Path,
    discovered_urls_output: Path,
    resume: bool,
    delay_seconds: float,
) -> pd.DataFrame:
    session = requests.Session()
    urls = read_event_urls(
        path=None,
        include_archive=False,
        ufcstats_events=events_path,
        session=session,
        delay=delay_seconds,
        min_date="2007-01-01",
        top_n=3,
        discovered_urls_output=discovered_urls_output,
    )
    if not urls:
        raise RuntimeError("No BestFightOdds event URLs were discovered.")

    completed_urls = read_completed_source_urls(odds_output) if resume else set()
    if odds_output.exists() and not resume:
        odds_output.unlink()

    for index, url in enumerate(urls, start=1):
        if url in completed_urls:
            print(f"[{index}/{len(urls)}] skipping completed {url}", flush=True)
            continue

        print(f"[{index}/{len(urls)}] {url}", flush=True)
        records = scrape_event_moneylines(url, session)
        if records:
            page = pd.DataFrame(records)
            odds_output.parent.mkdir(parents=True, exist_ok=True)
            page.to_csv(odds_output, mode="a", header=not odds_output.exists(), index=False)
            print(f"  wrote {len(records)} rows for {records[0]['event']}", flush=True)
        else:
            print("  no fight rows found", flush=True)

    return pd.read_csv(odds_output) if odds_output.exists() else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update fight results joined with BestFightOdds moneylines.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_OUTPUT)
    parser.add_argument("--odds", type=Path, default=DEFAULT_ODDS_OUTPUT)
    parser.add_argument("--discovered-urls-output", type=Path, default=DEFAULT_DISCOVERED_URLS_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-ufcstats", action="store_true", help="Update UFCStats source CSVs first.")
    parser.add_argument("--refresh-odds", action="store_true", help="Re-scrape BestFightOdds moneylines before joining.")
    parser.add_argument("--resume-odds", action="store_true", help="Resume an existing odds scrape.")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.refresh_ufcstats or not args.results.exists() or not args.events.exists():
        update_ufcstats_fight_data(dry_run=args.dry_run, backup=not args.no_backup)

    if args.refresh_odds or not args.odds.exists():
        if args.dry_run:
            print("Dry run selected; skipping BestFightOdds scrape.")
            return 0
        odds = ensure_bestfightodds_moneylines(
            args.odds,
            args.events,
            args.discovered_urls_output,
            resume=args.resume_odds,
            delay_seconds=args.delay,
        )
        if odds.empty:
            raise RuntimeError("BestFightOdds scrape did not produce any rows.")

    join_output = args.output
    temp_output: Path | None = None
    if args.dry_run:
        with NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            temp_output = Path(handle.name)
        join_output = temp_output

    joined = build_results_with_odds(args.results, args.events, args.odds, join_output)
    if args.dry_run:
        if temp_output and temp_output.exists():
            temp_output.unlink()
        print(f"Dry run selected; built shape would be {joined.shape}.")
        return 0

    # build_results_with_odds writes directly; rewrite atomically for consistent backups.
    write_csv_atomic(joined, args.output, backup=not args.no_backup)
    matched = int(joined["odds_matched"].sum())
    inferred = int(joined["odds_inferred"].sum())
    print(f"Updated {args.output} with shape {joined.shape}")
    print(f"Matched odds for {matched}/{len(joined)} fights; inferred odds for {inferred}/{len(joined)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
