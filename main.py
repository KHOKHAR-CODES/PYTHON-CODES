"""
main.py

Multi-File Keyword Search Tool
===============================
Searches every .txt file inside the documents/ folder for a keyword
the user types in, shows every matching line (filename, line number,
and text), and saves a timestamped report to results/search_results.txt.

Run it with:
    python main.py
"""

import os

from file_utils import get_text_files, read_lines, save_results_to_file
from text_utils import (
    search_lines_for_keyword,
    format_match_line,
    get_current_timestamp,
)

# Folder that holds the .txt files we search through
DOCUMENTS_FOLDER = "documents"

# File that stores a running history of every search performed
RESULTS_FILE = os.path.join("results", "search_results.txt")

SEPARATOR = "-" * 50


def run_single_search(keyword):
    """
    Search every .txt file in DOCUMENTS_FOLDER for keyword, print the
    results to the screen, and append a timestamped report to
    RESULTS_FILE. Returns nothing; all output happens via print()
    and the results file.
    """
    text_files = get_text_files(DOCUMENTS_FOLDER)

    # Handle a missing (or empty) documents folder without crashing
    if not text_files:
        print(f"\nNo .txt files found in the '{DOCUMENTS_FOLDER}' folder.")
        print("Make sure the folder exists and contains at least one .txt file.\n")
        return

    total_matches = 0
    files_with_match = 0
    match_lines = []  # formatted "filename (line N): text" strings

    for file_path in text_files:
        lines = read_lines(file_path)
        matches = search_lines_for_keyword(lines, keyword)

        if matches:
            files_with_match += 1
            display_name = os.path.basename(file_path)
            for line_number, line_text in matches:
                match_lines.append(format_match_line(display_name, line_number, line_text))
                total_matches += 1

    summary_lines = [
        f"Files searched   : {len(text_files)}",
        f"Files with match : {files_with_match}",
        f"Total matches    : {total_matches}",
    ]

    # ----- Show results on screen -----
    print("\n" + "=" * 50)
    if match_lines:
        for line in match_lines:
            print(line)
    else:
        print("No matches found.")
    print(SEPARATOR)
    for line in summary_lines:
        print(line)
    print("=" * 50 + "\n")

    # ----- Save the same results to file, with a timestamp -----
    timestamp = get_current_timestamp()
    report_lines = (
        [
            f"Search performed : {timestamp}",
            f"Keyword          : {keyword}",
            SEPARATOR,
        ]
        + (match_lines if match_lines else ["No matches found."])
        + [SEPARATOR]
        + summary_lines
    )

    save_results_to_file(RESULTS_FILE, report_lines)
    print(f"Results saved to '{RESULTS_FILE}'\n")


def main():
    """
    Main program loop: repeatedly ask the user for a keyword to
    search, running a new search each time, until they choose to exit.
    """
    print("Multi-File Keyword Search Tool")
    print(f"Searches all .txt files in the '{DOCUMENTS_FOLDER}' folder.\n")

    while True:
        keyword = input("Enter a keyword to search (or type 'exit' to quit): ").strip()

        if keyword.lower() == "exit":
            print("Goodbye!")
            break

        if not keyword:
            print("Please enter a non-empty keyword.\n")
            continue

        run_single_search(keyword)


if __name__ == "__main__":
    main()
