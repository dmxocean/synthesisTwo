# -*- coding: utf-8 -*-
"""
Provides network utilities and filesystem operations for PDF dataset acquisition

Primary inputs include remote URLs and chronological metadata for directory mapping. Design decisions focus on robust retry policies and filesystem collision avoidance
"""

import os
import re
from urllib.parse import unquote
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.core.config import PATH_DIR_RAW

def get_session():
    """
    Establish network session with configurable retry policies

    The session is configured with backoff strategies and status forcelists to handle intermittent network failures gracefully
    """
    session = requests.Session()
    retry = Retry(
        total=8,
        connect=5,  # Retry on ConnectTimeoutError or connection refused
        read=3,
        backoff_factor=2,  # Exponential backoff between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def sanitize_filename(filename):
    """
    Normalize filename strings by excluding unauthorized system characters

    This method replaces characters that are illegal across major filesystems with underscores to ensure cross-platform compatibility
    """
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", filename)
    return sanitized

def check_file_exists(url, year):
    """
    Verify if a specific PDF asset is already persisted in the local filesystem

    Args:
        url (str): Remote source URL
        year (int): Publication year
    Returns:
        bool: True if the file exists in the year-categorized directory
    """
    filename = unquote(url.split("/")[-1])
    filename = sanitize_filename(filename)
    file_path = os.path.join(PATH_DIR_RAW, str(year), filename)
    return os.path.exists(file_path)

def download_pdf(url, year, filename=None, session=None):
    """
    Retrieve external PDF assets and store them inside the chronological hierarchy

    Args:
        url (str): Remote source URL for the PDF document
        year (int): Publication year used for directory categorization
        filename (str, optional): Custom filename to override automatic extraction
        session (requests.Session, optional): Reusable session for optimized connections
    Returns:
        str: The local filename of the acquired asset or None if the operation fails
    """
    if not filename:
        filename = unquote(url.split("/")[-1])  # Recover original filename
        filename = sanitize_filename(filename)

    target_dir = os.path.join(PATH_DIR_RAW, str(year))  # Ensure isolation by year
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, filename)

    if os.path.exists(file_path):
        return filename

    req_session = session or get_session()  # Maintain connection pooling

    try:
        response = req_session.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filename

    except Exception as e:
        print(f"[!] Failed to acquire {filename}: {e}")  # Alert on network or IO failures
        return None
