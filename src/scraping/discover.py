# -*- coding: utf-8 -*-
"""
Reusable components for GUIRAD metadata discovery

This module provides the functional building blocks for the discovery workflow, including MARCXML parsing, paginated fetching from the DDD portal, and persistence of the metadata index. It is designed to be composed by external synchronization scripts
"""

import os
import json
import re
import xml.etree.ElementTree as ET

from src.scraping.download import check_file_exists
from src.core.config import PATH_FILE_JSON, PATH_FILE_REPORT

BASE_URL = "https://ddd.uab.cat"
SEARCH_URL = f"{BASE_URL}/search?cc=guirad&ln=es&of=xm&rg=100"
MARC_NS = {"marc": "http://www.loc.gov/MARC21/slim"}
PAGE_SIZE = 100

EPOCHS = [
    {"name": "Monarchy & Dictatorship (1924-1930)", "range": range(1924, 1931)},
    {"name": "Second Spanish Republic (1931-1935)", "range": range(1931, 1936)},
    {"name": "Spanish Civil War (1936-1938)", "range": range(1936, 1939)},
    {"name": "Francoist Regime (1939-1953)", "range": range(1939, 1954)},
]

def resolve_url(base, url):
    """
    Normalize relative URLs into absolute endpoints

    Args:
        base (str): Base host URL
        url (str): Relative or absolute URL string
    Returns:
        str: Fully qualified absolute URL
    """
    return url if "://" in url else f"{base.rstrip('/')}/{url.lstrip('/')}"

def extract_languages(record_xml, namespaces=MARC_NS):
    """
    Extract and normalize MARC language codes from 041$a subfields

    The function scans for three-letter ISO codes and returns a sorted list of unique language identifiers found within the record
    """
    languages = set()
    lang_elems = record_xml.findall("./marc:datafield[@tag='041']/marc:subfield[@code='a']", namespaces)

    for elem in lang_elems:
        if not elem.text:
            continue
        normalized = elem.text.strip().lower()
        if not normalized:
            continue
        codes = re.findall(r"[a-z]{3}", normalized)
        if codes:
            languages.update(codes)
        else:
            languages.add(normalized)

    return sorted(languages)

def parse_xml_record(record_xml):
    """
    Parse MARCXML control and data fields into a structured dictionary

    This extractor retrieves unique record identifiers, titles, publication years, and associated PDF asset links while performing year recovery from titles if necessary
    """
    recid_elem = record_xml.find("./marc:controlfield[@tag='001']", MARC_NS)
    recid = recid_elem.text if recid_elem is not None else "unknown"

    title = "unknown"
    title_elem = record_xml.find("./marc:datafield[@tag='245']/marc:subfield[@code='a']", MARC_NS)
    if title_elem is not None:
        title = title_elem.text

    year = "unknown"
    year_elem = record_xml.find("./marc:datafield[@tag='260']/marc:subfield[@code='c']", MARC_NS)
    if year_elem is not None:
        year_match = re.search(r"(19\d{2})", year_elem.text)
        if year_match:
            year = year_match.group(1)

    if year == "unknown":  # Attempt recovery from title string
        year_match = re.search(r"(19\d{2})", title)
        year = year_match.group(1) if year_match else "unknown"

    languages = extract_languages(record_xml)

    pdf_links = []
    link_elems = record_xml.findall("./marc:datafield[@tag='856']/marc:subfield[@code='u']", MARC_NS)
    for elem in link_elems:
        url = elem.text
        if url and ".pdf" in url.lower() and "guiradbcn" in url.lower():
            pdf_links.append(resolve_url(BASE_URL, url))

    return {
        "id": recid,
        "url": f"{BASE_URL}/record/{recid}?ln=es",
        "year": year,
        "title": title,
        "languages": languages,
        "pdf_links": pdf_links,
    }

def iter_search_pages(session, search_query, page_size=PAGE_SIZE):
    """
    Yield parsed records from paginated DDD MARCXML endpoints

    The generator iterates through result pages using a cursor and yields lists of processed records until no further items are returned
    Args:
        session (requests.Session): Pooled session with retry policy
        search_query (str): Base query URL without the cursor
        page_size (int): Records per page
    Yields:
        list: Collection of parsed record dictionaries for one page
    """
    jrec = 1
    while True:
        url = f"{search_query}&jrec={jrec}"
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            records = root.findall(".//{http://www.loc.gov/MARC21/slim}record")
        except Exception as e:
            print(f"[!] Search fetch logic fault: {e}")  # Report network or parsing failures
            return
        if not records:
            return
        yield [parse_xml_record(r) for r in records]
        jrec += page_size

def load_database_state(path=PATH_FILE_JSON):
    """
    Load the metadata index from a JSON file

    Args:
        path (str): Path to the JSON database file
    Returns:
        dict: Metadata index keyed by record identifier
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {item["id"]: item for item in json.load(f)}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}  # Return empty index on corruption or absence

def save_database_state(database, path=PATH_FILE_JSON):
    """
    Serialize the metadata index to persistent JSON storage

    This method ensures that the current discovery state is saved to disk for crash recovery and subsequent acquisition runs
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(database.values()), f, indent=2, ensure_ascii=False)

def generate_report(records, path=PATH_FILE_REPORT):
    """
    Synthesize a statistical report of the dataset discovery

    The report provides a global summary followed by a breakdown of records and PDF acquisition status categorized by historical epochs
    Args:
        records (list): Record dictionaries discovered in the current run
        path (str): Destination path for the report file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("GUIRAD DATASET ANALYSIS REPORT\n")
        f.write("(Current --info execution, full-year scan)\n\n")

        total_identified = sum(len(r["pdf_links"]) for r in records)
        total_acquired = sum(1 for r in records for url in r["pdf_links"] if check_file_exists(url, r["year"]))

        f.write("Global Summary:\n")
        f.write(f"[*] Total Years Mapped: {len(records)}\n")
        f.write(f"[*] Total PDFs Identified: {total_identified}\n")
        f.write(f"[*] Total PDFs Acquired: {total_acquired}\n\n")
        f.write("Epoch Breakdown\n\n")

        for epoch in EPOCHS:
            epoch_data = [r for r in records
                          if str(r["year"]).isdigit() and int(r["year"]) in epoch["range"]]
            pdfs_identified = sum(len(r["pdf_links"]) for r in epoch_data)
            pdfs_acquired = sum(1 for r in epoch_data for url in r["pdf_links"]
                                if check_file_exists(url, r["year"]))
            langs = sorted({l for r in epoch_data for l in r["languages"]})

            f.write(f"\n[{epoch['name']}]\n")
            f.write(f" > Records: {len(epoch_data)}\n")
            f.write(f" > PDFs identified: {pdfs_identified}\n")
            f.write(f" > PDFs acquired: {pdfs_acquired}\n")
            f.write(f" > Languages: {', '.join(langs) if langs else 'None'}\n")

    print(f"[*] Report generated at {path}")  # Log successful report creation
