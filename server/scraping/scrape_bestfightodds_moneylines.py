"""
Scrape fight-level moneyline odds from BestFightOdds event pages.

This scraper outputs a normalized CSV that can be passed to
build_historical_fight_features.py with --odds-csv.

BestFightOdds event URLs look like:
https://www.bestfightodds.com/events/ufc-vegas-116-4147
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scraping_common import CACHE_DIR


BASE_URL = "https://www.bestfightodds.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}
BFO_ROTATION_CHARS = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"


def is_ufc_event_url(url: str) -> bool:
    return "/events/ufc" in url.lower()


def get_with_retries(session: requests.Session, url: str, retries: int = 3, timeout: int = 30) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"{response.status_code} retryable status", response=response)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"skipping {url}: {exc}")
                return None
            time.sleep(2 * attempt)
    return None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def event_search_terms(event_name: str) -> list[str]:
    event_name = clean_text(event_name)
    terms = []
    if ":" in event_name:
        prefix, suffix = event_name.split(":", 1)
        prefix = prefix.strip()
        suffix = suffix.replace("vs.", "vs").strip()
        terms.append(suffix)
        terms.append(f"{prefix} {suffix}")
        if normalize_event_text(prefix) != "ufc fight night":
            terms.append(prefix)
    terms.append(event_name)
    number = re.search(r"\bUFC\s+\d+\b", event_name, flags=re.I)
    if number:
        terms.insert(0, number.group(0))
    return list(dict.fromkeys(term for term in terms if term))


def normalize_event_text(value: str) -> str:
    value = clean_text(value).lower().replace(".", "")
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return clean_text(value)


def token_score(query: str, candidate: str) -> int:
    query_tokens = set(normalize_event_text(query).split())
    candidate_tokens = set(normalize_event_text(candidate).split())
    return len(query_tokens & candidate_tokens)


def parse_american_odds(value: str) -> int | None:
    value = clean_text(value)
    match = re.fullmatch(r"[+-]\d+", value)
    if not match:
        return None
    return int(value)


def decimal_to_american(decimal_odds: float) -> int | None:
    if decimal_odds is None or pd.isna(decimal_odds) or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def american_to_implied_probability(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_probability_to_american(probability: float) -> int:
    probability = min(max(probability, 0.001), 0.999)
    if probability >= 0.5:
        return int(round(-(probability / (1.0 - probability)) * 100.0))
    return int(round(((1.0 - probability) / probability) * 100.0))


def consensus_american_odds(odds: list[int]) -> int:
    probability = sum(american_to_implied_probability(value) for value in odds) / len(odds)
    return implied_probability_to_american(probability)


def decode_bfo_payload(value: str) -> str:
    decoded = base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", value)).decode("utf-8")
    half = len(BFO_ROTATION_CHARS) // 2
    output = []
    for char in decoded:
        index = BFO_ROTATION_CHARS.find(char)
        output.append(BFO_ROTATION_CHARS[(index + half) % len(BFO_ROTATION_CHARS)] if index >= 0 else char)
    return "".join(output)


def extract_mean_history_lines(matchup_id: str, participant: int, session: requests.Session, referer: str) -> dict[str, object]:
    response = get_with_retries(session, f"{BASE_URL}/api/ggd?m={matchup_id}&p={participant}")
    if response is None:
        return {
            "open": None,
            "close": None,
            "open_ts": None,
            "close_ts": None,
            "history_points": 0,
        }

    try:
        payload = json.loads(decode_bfo_payload(response.text))
    except (ValueError, json.JSONDecodeError):
        return {
            "open": None,
            "close": None,
            "open_ts": None,
            "close_ts": None,
            "history_points": 0,
        }

    if not payload or not payload[0].get("data"):
        return {
            "open": None,
            "close": None,
            "open_ts": None,
            "close_ts": None,
            "history_points": 0,
        }

    points = payload[0]["data"]
    first = points[0]
    last = points[-1]
    return {
        "open": decimal_to_american(float(first.get("y"))),
        "close": decimal_to_american(float(last.get("y"))),
        "open_ts": pd.to_datetime(first.get("x"), unit="ms", utc=True).isoformat() if first.get("x") else None,
        "close_ts": pd.to_datetime(last.get("x"), unit="ms", utc=True).isoformat() if last.get("x") else None,
        "history_points": len(points),
    }


def extract_event_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    meta = soup.find("meta", {"name": "description"})
    content = meta.get("content", "") if meta else ""
    match = re.search(r"on ([A-Z][a-z]+ \d{1,2}, \d{4})", content)
    if not match:
        return None
    date = pd.to_datetime(match.group(1), errors="coerce")
    return None if pd.isna(date) else date


def extract_event_name(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))
    title = soup.find("title")
    return clean_text(title.get_text(" ", strip=True).split(" odds:")[0]) if title else ""


def row_values(row) -> list[str]:
    values = []
    for cell in row.find_all(["td", "th"]):
        text = clean_text(cell.get_text(" ", strip=True))
        if text in {"", "▲", "▼", "n/a"}:
            continue
        values.append(text)
    return values


def parse_matchup_id(row) -> str | None:
    matchup_link = row.find("a", class_="bfo-admin-link")
    if matchup_link:
        match = re.search(r"/matchups/(\d+)", matchup_link.get("href", ""))
        if match:
            return match.group(1)

    line_history_cell = row.find(attrs={"data-li": True})
    if line_history_cell:
        match = re.search(r"\[(?:1|2),\s*(\d+)\]", line_history_cell.get("data-li", ""))
        if match:
            return match.group(1)

    row_id = row.get("id", "")
    match = re.search(r"mu-(\d+)", row_id)
    return match.group(1) if match else None


def parse_fighter_row(row) -> dict[str, object] | None:
    fighter_tag = row.find("span", class_="t-b-fcc")
    if not fighter_tag:
        return None

    fighter = clean_text(fighter_tag.get_text(" ", strip=True))
    odds = []
    for cell in row.find_all("td"):
        classes = cell.get("class") or []
        if "but-sg" not in classes:
            continue
        parsed = parse_american_odds(clean_text(cell.get_text(" ", strip=True)).split(" ")[0])
        if parsed is not None and 100 <= abs(parsed) <= 2000:
            odds.append(parsed)

    matchup_id = parse_matchup_id(row)
    if not odds and not matchup_id:
        return None

    return {
        "fighter": fighter,
        "odds": odds,
        "matchup_id": matchup_id,
    }


def scrape_event_moneylines(url: str, session: requests.Session) -> list[dict[str, object]]:
    response = get_with_retries(session, url)
    if response is None:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    event_name = extract_event_name(soup)
    event_date = extract_event_date(soup)
    tables = soup.find_all("table", class_="odds-table")

    if len(tables) < 2:
        return []

    fighter_rows = []
    for row in tables[1].find_all("tr"):
        classes = row.get("class") or []
        if "pr" in classes:
            continue
        parsed = parse_fighter_row(row)
        if parsed:
            fighter_rows.append(parsed)

    records = []
    for i in range(0, len(fighter_rows) - 1, 2):
        fighter1 = fighter_rows[i]["fighter"]
        odds1 = fighter_rows[i]["odds"]
        matchup_id = fighter_rows[i].get("matchup_id")
        fighter2 = fighter_rows[i + 1]["fighter"]
        odds2 = fighter_rows[i + 1]["odds"]

        f1_history = extract_mean_history_lines(matchup_id, 1, session, url) if matchup_id else {}
        f2_history = extract_mean_history_lines(matchup_id, 2, session, url) if matchup_id else {}
        f1_close = f1_history.get("close") or (consensus_american_odds(odds1) if odds1 else None)
        f2_close = f2_history.get("close") or (consensus_american_odds(odds2) if odds2 else None)
        records.append(
            {
                "event": event_name,
                "date": event_date.date().isoformat() if event_date is not None else "",
                "fighter1": fighter1,
                "fighter2": fighter2,
                "f1_open": f1_history.get("open"),
                "f2_open": f2_history.get("open"),
                "f1_close": f1_close,
                "f2_close": f2_close,
                "f1_odds": f1_close,
                "f2_odds": f2_close,
                "f1_open_ts": f1_history.get("open_ts"),
                "f2_open_ts": f2_history.get("open_ts"),
                "f1_close_ts": f1_history.get("close_ts"),
                "f2_close_ts": f2_history.get("close_ts"),
                "f1_history_points": f1_history.get("history_points", 0),
                "f2_history_points": f2_history.get("history_points", 0),
                "f1_books": len(odds1),
                "f2_books": len(odds2),
                "matchup_id": matchup_id,
                "source_url": url,
            }
        )

    return records


def archive_ufc_event_urls(session: requests.Session) -> list[str]:
    response = get_with_retries(session, f"{BASE_URL}/archive")
    if response is None:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for link in soup.find_all("a", href=True):
        text = clean_text(link.get_text(" ", strip=True))
        href = link["href"]
        if text.upper().startswith("UFC") and href.startswith("/events/"):
            urls.append(urljoin(BASE_URL, href))
    return list(dict.fromkeys(urls))


def search_event_urls(query: str, session: requests.Session, top_n: int = 3) -> list[str]:
    response = get_with_retries(session, f"{BASE_URL}/search?query={quote_plus(query)}")
    if response is None:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    candidates = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.startswith("/events/"):
            continue
        text = clean_text(link.get_text(" ", strip=True))
        candidates.append((token_score(query, text), text, urljoin(BASE_URL, href)))

    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    return [url for score, _, url in candidates[:top_n] if score > 0]


def discover_ufcstats_event_urls(
    ufcstats_events: Path,
    session: requests.Session,
    delay: float,
    min_date: str | None = "2007-01-01",
    top_n: int = 3,
    discovered_urls_output: Path | None = None,
) -> list[str]:
    events = pd.read_csv(ufcstats_events)
    events["event_date"] = pd.to_datetime(events["DATE"], errors="coerce", format="mixed")
    if min_date:
        events = events[events["event_date"] >= pd.to_datetime(min_date)]

    urls = []
    for index, row in events.sort_values("event_date", ascending=False).iterrows():
        event_name = row["EVENT"]
        discovered_for_event = []
        for term in event_search_terms(event_name):
            discovered_for_event.extend(search_event_urls(term, session, top_n=top_n))
            time.sleep(delay)
            if discovered_for_event:
                break
        urls.extend(discovered_for_event)
        unique_urls = list(dict.fromkeys(urls))
        if discovered_urls_output:
            discovered_urls_output.write_text("\n".join(unique_urls) + "\n")
        print(f"discovered {len(discovered_for_event):>2} BFO candidates for {event_name}", flush=True)

    return list(dict.fromkeys(urls))


def read_event_urls(
    path: Path | None,
    include_archive: bool,
    ufcstats_events: Path | None,
    session: requests.Session,
    delay: float,
    min_date: str | None,
    top_n: int,
    discovered_urls_output: Path | None,
) -> list[str]:
    urls = []
    if include_archive:
        urls.extend(archive_ufc_event_urls(session))
    if ufcstats_events:
        urls.extend(
            discover_ufcstats_event_urls(
                ufcstats_events,
                session,
                delay,
                min_date=min_date,
                top_n=top_n,
                discovered_urls_output=discovered_urls_output,
            )
        )
    if path:
        urls.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    return list(dict.fromkeys(url for url in urls if is_ufc_event_url(url)))


def read_completed_source_urls(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    try:
        existing = pd.read_csv(output, usecols=["source_url"])
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(existing["source_url"].dropna().astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-urls", type=Path, default=None, help="Text file with one BestFightOdds event URL per line.")
    parser.add_argument("--archive", action="store_true", help="Also scrape currently listed UFC links from /archive.")
    parser.add_argument("--ufcstats-events", type=Path, default=None, help="Discover BFO event URLs from a UFCStats event_details CSV.")
    parser.add_argument("--min-date", type=str, default="2007-01-01")
    parser.add_argument("--search-top-n", type=int, default=3)
    parser.add_argument("--discovered-urls-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=CACHE_DIR / "bestfightodds_moneylines.csv")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Skip event URLs already present in the output CSV.")
    args = parser.parse_args()

    session = requests.Session()
    urls = read_event_urls(
        args.event_urls,
        args.archive,
        args.ufcstats_events,
        session,
        args.delay,
        args.min_date,
        args.search_top_n,
        args.discovered_urls_output,
    )
    if not urls:
        raise SystemExit("No event URLs provided. Use --event-urls or --archive.")

    all_records = []
    completed_urls = read_completed_source_urls(args.output) if args.resume else set()
    if args.resume and completed_urls:
        print(f"resume enabled: skipping {len(completed_urls)} completed event URLs from {args.output}", flush=True)
    if args.output.exists() and not args.resume:
        args.output.unlink()

    for index, url in enumerate(urls, start=1):
        if url in completed_urls:
            print(f"[{index}/{len(urls)}] skipping completed {url}", flush=True)
            continue
        print(f"[{index}/{len(urls)}] {url}", flush=True)
        records = scrape_event_moneylines(url, session)
        all_records.extend(records)
        if records:
            df_page = pd.DataFrame(records)
            df_page.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
            print(f"  wrote {len(records)} fight rows for {records[0]['event']}", flush=True)
        else:
            print("  no fight rows found", flush=True)
        time.sleep(args.delay)

    df = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame(all_records)
    if not args.output.exists():
        df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {df.shape}", flush=True)


if __name__ == "__main__":
    main()
