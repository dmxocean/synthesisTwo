# -*- coding: utf-8 -*-
"""
GUIRAD discovery and asset synchronization executable workflow

This script synchronizes assets from the DDD archive by walking MARCXML search results, maintaining an incremental metadata index, and downloading missing PDFs. It supports targeted year ranges, parallel downloads, and metadata-only audit reports
"""

import os
import re
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.config import PATH_DIR_ANALYSIS, PATH_FILE_JSON, PATH_FILE_REPORT
from src.scraping.download import get_session, download_pdf
from src.scraping.discover import (
    SEARCH_URL,
    iter_search_pages,
    load_database_state,
    save_database_state,
    generate_report,
)

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_years(value):
    """
    Parse a year specification string into a list of integer years

    Supported formats include single years, comma-separated ranges, and lists of tuples
    """
    value = value.strip()

    # List of ranges: [(a,b),(c,d),...]
    if value.startswith("["):
        pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", value)
        if not pairs:
            raise argparse.ArgumentTypeError(f"Cannot parse year list: {value!r}")
        years = []
        for s, e in pairs:
            years.extend(range(int(s), int(e) + 1))
        return years

    # Single range: a,b or (a,b)
    pair = re.match(r"^\(?(\d+)\s*,\s*(\d+)\)?$", value)
    if pair:
        return list(range(int(pair.group(1)), int(pair.group(2)) + 1))

    # Single year
    if value.isdigit():
        return [int(value)]

    raise argparse.ArgumentTypeError(
        f"Invalid --year format: {value!r}\n"
        "  Use: 1934  |  1924,1953  |  (1924,1953)  |  [(1924,1930),(1936,1938)]"
    )


def discover(session, target_year=None, info_only=False):
    """
    Traverse search results and return discovered archival records
    """
    database = {} if info_only else load_database_state()

    search_query = SEARCH_URL
    if not info_only and target_year:
        search_query += f"&p={target_year}"

    active_records, seen_ids = [], set()

    for page in iter_search_pages(session, search_query):
        new_in_page = False
        for meta in page:
            rid = meta["id"]

            if not info_only:
                if rid not in database:
                    database[rid] = meta
                    save_database_state(database)
                else:
                    meta = database[rid]

            if not info_only and target_year and str(meta["year"]) != str(target_year):
                continue

            if rid not in seen_ids:
                seen_ids.add(rid)
                active_records.append(meta)
                new_in_page = True
                print(f"Record {rid} ({meta['year']}) indexed successfully")  # Log record indexing

        if not new_in_page:
            break

    if not info_only:
        save_database_state(database)
    return active_records


_thread_local = threading.local()


def _thread_session():
    """
    Return a per-thread requests session
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = get_session()
    return _thread_local.session


def _download_one(args):
    """
    Worker function to download a single PDF asset
    """
    url, year = args
    time.sleep(random.uniform(0.5, 1.5))
    return download_pdf(url, year, session=_thread_session())


def download_assets(records, workers=4):
    """
    Download PDF assets for the given records concurrently
    """
    tasks = [(url, record["year"]) for record in records for url in record["pdf_links"]]
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, task): task for task in tasks}
        for future in as_completed(futures):
            filename = future.result()
            if filename:
                total += 1
                print(f"File {filename} acquired")  # Log successful download
    return total


def main():
    """
    Entry point for the GUIRAD asset synchronizer
    """
    parser = argparse.ArgumentParser(
        description="Guirad Dataset Architect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=parse_years,
        metavar="YEAR_SPEC",
        help=(
            "Year(s) to process. Formats: "
            "1934 | 1924,1953 | (1924,1953) | [(1924,1930),(1936,1938)]"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Metadata-only audit - no downloads, no index writes",
    )
    args = parser.parse_args()

    os.makedirs(PATH_DIR_ANALYSIS, exist_ok=True)
    session = get_session()

    if args.info:
        print("Info Mode: full metadata scan, no downloads, no guirad.json updates")  # Log audit start
        records = discover(session, target_year=None, info_only=True)
        generate_report(records)
        print(f"Info report synchronized at: {PATH_FILE_REPORT}")  # Log report location
        return

    years = args.year or [None]
    print(f"Download Mode (years={args.year or 'all'})")  # Log synchronization start

    total_acquired = 0
    for year in years:
        if year:
            print(f"\nProcessing Year: {year}")  # Log active year
        records = discover(session, target_year=year, info_only=False)
        print("Sync Phase Active")  # Log start of download phase
        count = download_assets(records, workers=args.workers)
        total_acquired += count
        if year:
            print(f"Year {year} complete - {count} files acquired")  # Log year completion

    print(f"\nSync complete - {total_acquired} files acquired | index at {PATH_FILE_JSON}")  # Log overall status


if __name__ == "__main__":
    main()
