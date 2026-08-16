"""
text_utils.py

Text searching and formatting helper functions for the
Multi-File Keyword Search Tool.

This module knows how to search inside a list of lines and how to
format matches for display / saving. It does not touch the
filesystem directly — that is file_utils.py's job.
"""

from datetime import datetime


def line_contains_keyword(line, keyword):
    """
    Case-insensitive check for whether keyword appears anywhere in line.
    Both strings are lower-cased before comparing, so "Python",
    "PYTHON", and "python" are all treated as the same keyword.
    """
    return keyword.lower() in line.lower()


def search_lines_for_keyword(lines, keyword):
    """
    Search a list of lines for keyword (case-insensitive).

    Returns a list of (line_number, line_text) tuples for every line
    that contains a match. Line numbers start at 1, matching how a
    text editor would number the lines.
    """
    matches = []
    for line_number, line_text in enumerate(lines, start=1):
        if line_contains_keyword(line_text, keyword):
            matches.append((line_number, line_text.strip()))
    return matches


def format_match_line(filename, line_number, line_text):
    """
    Format a single match for display on screen and for saving to
    the results file.

    Example output:
        notes.txt (line 4): Remember to water the plants
    """
    return f"{filename} (line {line_number}): {line_text}"


def get_current_timestamp():
    """
    Return the current date and time as a readable string,
    e.g. "2026-08-17 14:32:05". Used to timestamp each saved search.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
