"""
file_utils.py

File and folder helper functions for the Multi-File Keyword Search Tool.

This module is only responsible for talking to the filesystem:
finding .txt files, reading them safely, and saving the results
report. It does not know anything about *how* searching works —
that logic lives in text_utils.py.
"""

import os


def get_text_files(folder_path):
    """
    Look inside folder_path and return a sorted list of full paths
    to every .txt file found there.

    If folder_path does not exist (or is not a folder), this returns
    an empty list instead of raising an error, so the caller can
    handle a missing "documents" folder gracefully.
    """
    if not os.path.isdir(folder_path):
        return []

    text_file_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            text_file_paths.append(full_path)

    return sorted(text_file_paths)


def read_lines(file_path):
    """
    Read a text file and return its contents as a list of lines
    (with the trailing newline characters removed).

    If the file cannot be opened for some reason (permissions,
    bad encoding, etc.), a warning is printed and an empty list
    is returned so the rest of the program can keep running.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read().splitlines()
    except OSError as error:
        print(f"Warning: could not read '{file_path}' ({error})")
        return []


def ensure_folder_exists(folder_path):
    """
    Make sure folder_path exists, creating it (and any missing
    parent folders) if necessary. Does nothing if it already exists.
    """
    if folder_path:  # avoid calling makedirs("") when path has no folder part
        os.makedirs(folder_path, exist_ok=True)


def save_results_to_file(output_path, report_lines):
    """
    Append report_lines (a list of strings) to output_path, one
    search report at a time. Using append mode means every search
    performed during the program's lifetime (and across runs) is
    kept as a running history in results/search_results.txt.

    The results folder is created automatically if it is missing.
    """
    folder = os.path.dirname(output_path)
    ensure_folder_exists(folder)

    with open(output_path, "a", encoding="utf-8") as file:
        for line in report_lines:
            file.write(line + "\n")
        file.write("\n")  # blank line to separate this search from the next
