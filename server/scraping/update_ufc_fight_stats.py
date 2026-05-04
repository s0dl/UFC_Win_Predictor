"""
Update UFCStats event, fight-result, and round-stat CSVs.

The current model directly uses server/models/data/ufc_fight_stats.csv. The
event and result CSVs are kept in this folder because they are required inputs
for the odds join that builds ufc_fight_results_with_odds.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import scrape_ufc_stats_library as lib
from scraping_common import CACHE_DIR, DATA_DIR, read_csv_or_empty, write_csv_atomic


EVENT_DETAILS_COLUMNS = ["EVENT", "URL", "DATE", "LOCATION"]
FIGHT_DETAILS_COLUMNS = ["EVENT", "BOUT", "URL"]
FIGHT_RESULTS_COLUMNS = [
    "EVENT",
    "BOUT",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
    "REFEREE",
    "DETAILS",
    "URL",
]
TOTALS_COLUMNS = [
    "ROUND",
    "FIGHTER",
    "KD",
    "SIG.STR.",
    "SIG.STR. %",
    "TOTAL STR.",
    "TD",
    "TD %",
    "SUB.ATT",
    "REV.",
    "CTRL",
]
SIGNIFICANT_STRIKES_COLUMNS = [
    "ROUND",
    "FIGHTER",
    "SIG.STR.",
    "SIG.STR. %",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
]
FIGHT_STATS_COLUMNS = [
    "EVENT",
    "BOUT",
    "ROUND",
    "FIGHTER",
    "KD",
    "SIG.STR.",
    "SIG.STR. %",
    "TOTAL STR.",
    "TD",
    "TD %",
    "SUB.ATT",
    "REV.",
    "CTRL",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
]

DEFAULT_EVENTS_OUTPUT = CACHE_DIR / "ufc_event_details.csv"
DEFAULT_FIGHT_DETAILS_OUTPUT = CACHE_DIR / "ufc_fight_details.csv"
DEFAULT_RESULTS_OUTPUT = CACHE_DIR / "ufc_fight_results.csv"
DEFAULT_SCRAPING_STATS_OUTPUT = CACHE_DIR / "ufc_fight_stats.csv"
DEFAULT_DATA_STATS_OUTPUT = DATA_DIR / "ufc_fight_stats.csv"
DEFAULT_COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
DEFAULT_DELAY_SECONDS = 0.15


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def update_ufcstats_fight_data(
    events_output: Path = DEFAULT_EVENTS_OUTPUT,
    fight_details_output: Path = DEFAULT_FIGHT_DETAILS_OUTPUT,
    results_output: Path = DEFAULT_RESULTS_OUTPUT,
    scraping_stats_output: Path = DEFAULT_SCRAPING_STATS_OUTPUT,
    data_stats_output: Path = DEFAULT_DATA_STATS_OUTPUT,
    completed_events_url: str = DEFAULT_COMPLETED_EVENTS_URL,
    full_refresh: bool = False,
    backup: bool = True,
    dry_run: bool = False,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed_events = read_csv_or_empty(events_output, EVENT_DETAILS_COLUMNS)
    parsed_fight_details = read_csv_or_empty(fight_details_output, FIGHT_DETAILS_COLUMNS)
    parsed_results = read_csv_or_empty(results_output, FIGHT_RESULTS_COLUMNS)
    parsed_stats = read_csv_or_empty(data_stats_output, FIGHT_STATS_COLUMNS)

    updated_events = lib.parse_event_details(lib.get_soup(completed_events_url))
    known_events = set(parsed_events["EVENT"]) if not parsed_events.empty else set()
    complete_events = set(parsed_fight_details["EVENT"].dropna().unique()) if not parsed_fight_details.empty else set()
    all_events = list(updated_events["EVENT"])

    if full_refresh or parsed_results.empty or parsed_stats.empty:
        events_to_parse = all_events
    else:
        new_events = [event for event in all_events if event not in known_events]
        incomplete_events = [event for event in known_events if event not in complete_events]
        events_to_parse = new_events + incomplete_events

    if not events_to_parse:
        print("All UFCStats fight data is already current.")
        if scraping_stats_output != data_stats_output and not scraping_stats_output.exists() and not dry_run:
            write_csv_atomic(parsed_stats, scraping_stats_output, backup=backup)
        return updated_events, parsed_fight_details, parsed_results, parsed_stats

    event_urls = list(updated_events.loc[updated_events["EVENT"].isin(events_to_parse), "URL"])
    fight_detail_frames: list[pd.DataFrame] = []
    result_frames: list[pd.DataFrame] = []
    stat_frames: list[pd.DataFrame] = []

    for event_url in tqdm(event_urls, desc="Scraping UFCStats events"):
        event_soup = lib.get_soup(event_url)
        fight_details = lib.parse_fight_details(event_soup)
        fight_detail_frames.append(fight_details)

        for fight_url in tqdm(list(fight_details["URL"]), desc="Scraping fights", leave=False):
            fight_soup = lib.get_soup(fight_url)
            fight_results, fight_stats = lib.parse_organise_fight_results_and_stats(
                fight_soup,
                fight_url,
                FIGHT_RESULTS_COLUMNS,
                TOTALS_COLUMNS,
                SIGNIFICANT_STRIKES_COLUMNS,
            )
            result_frames.append(fight_results)
            stat_frames.append(fight_stats)
            if delay_seconds:
                time.sleep(delay_seconds)

    new_fight_details = _concat_or_empty(fight_detail_frames, FIGHT_DETAILS_COLUMNS)
    new_results = _concat_or_empty(result_frames, FIGHT_RESULTS_COLUMNS)
    new_stats = _concat_or_empty(stat_frames, FIGHT_STATS_COLUMNS)

    if full_refresh or parsed_results.empty or parsed_stats.empty:
        updated_fight_details = new_fight_details
        updated_results = new_results
        updated_stats = new_stats
    else:
        parsed_fight_details = parsed_fight_details[~parsed_fight_details["EVENT"].isin(events_to_parse)]
        parsed_results = parsed_results[~parsed_results["EVENT"].isin(events_to_parse)]
        parsed_stats = parsed_stats[~parsed_stats["EVENT"].isin(events_to_parse)]
        updated_fight_details = pd.concat([new_fight_details, parsed_fight_details], ignore_index=True)
        updated_results = pd.concat([new_results, parsed_results], ignore_index=True)
        updated_stats = pd.concat([new_stats, parsed_stats], ignore_index=True)

    updated_events = updated_events[EVENT_DETAILS_COLUMNS]
    updated_fight_details = updated_fight_details[FIGHT_DETAILS_COLUMNS]
    updated_results = updated_results[FIGHT_RESULTS_COLUMNS]
    updated_stats = updated_stats[FIGHT_STATS_COLUMNS]

    print(
        f"Events parsed: {len(events_to_parse)}; "
        f"fight result rows: {len(updated_results)}; stat rows: {len(updated_stats)}."
    )

    if not dry_run:
        write_csv_atomic(updated_events, events_output, backup=backup)
        write_csv_atomic(updated_fight_details, fight_details_output, backup=backup)
        write_csv_atomic(updated_results, results_output, backup=backup)
        write_csv_atomic(updated_stats, data_stats_output, backup=backup)
        write_csv_atomic(updated_stats, scraping_stats_output, backup=backup)

    return updated_events, updated_fight_details, updated_results, updated_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update UFCStats fight stats CSVs.")
    parser.add_argument("--events-output", type=Path, default=DEFAULT_EVENTS_OUTPUT)
    parser.add_argument("--fight-details-output", type=Path, default=DEFAULT_FIGHT_DETAILS_OUTPUT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--scraping-stats-output", type=Path, default=DEFAULT_SCRAPING_STATS_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_STATS_OUTPUT)
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    update_ufcstats_fight_data(
        events_output=args.events_output,
        fight_details_output=args.fight_details_output,
        results_output=args.results_output,
        scraping_stats_output=args.scraping_stats_output,
        data_stats_output=args.output,
        full_refresh=args.full_refresh,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        delay_seconds=args.delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
