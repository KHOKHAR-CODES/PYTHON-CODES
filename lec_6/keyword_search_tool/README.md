# Multi-File Keyword Search Tool

A simple command-line tool that searches every `.txt` file in a
`documents/` folder for a keyword you type in, shows every matching
line, and saves a timestamped report to `results/search_results.txt`.

## Project structure

```
keyword_search_tool/
├── main.py                    # Program entry point / user interaction loop
├── text_utils.py               # Custom module: searching & formatting helpers
├── file_utils.py                # Custom module: file & folder helpers
├── requirements.txt            # No third-party dependencies needed
├── README.md
├── documents/                   # Put the .txt files you want to search here
│   ├── notes.txt
│   ├── todo.txt
│   └── story.txt
└── results/
    └── search_results.txt       # Created automatically the first time you run a search
```

## Requirements

- Python 3.4 or later
- No external packages (see `requirements.txt`) — only the standard
  library (`os`, `datetime`) is used.

## How to run

From inside the `keyword_search_tool` folder:

```bash
python main.py
```

You'll be prompted to enter a keyword. The tool will:

1. Look through every `.txt` file in `documents/`.
2. Find every line that contains your keyword (case-insensitive —
   searching "python" also matches "Python" and "PYTHON").
3. Print each match on screen as `filename (line N): matching text`.
4. Print a summary: how many files were searched, how many files
   contained a match, and the total number of matches.
5. Append the same report — with a timestamp — to
   `results/search_results.txt`.

You can keep entering new keywords to run as many searches as you
like. Type `exit` when you're done.

### Example session

```
Multi-File Keyword Search Tool
Searches all .txt files in the 'documents' folder.

Enter a keyword to search (or type 'exit' to quit): python

==================================================
notes.txt (line 3): Today we discussed the new Python project for the search tool.
notes.txt (line 4): The team agreed that Python was a good choice because it's easy to read.
story.txt (line 3): Once upon a time, a curious programmer decided to learn python programming.
todo.txt (line 3): 1. Finish writing the PYTHON search functions in text_utils.py
--------------------------------------------------
Files searched   : 3
Files with match : 3
Total matches    : 4
==================================================

Results saved to 'results/search_results.txt'

Enter a keyword to search (or type 'exit' to quit): exit
Goodbye!
```

## Modules

- **`file_utils.py`** — everything that touches the filesystem:
  finding `.txt` files in `documents/`, reading them safely, making
  sure the `results/` folder exists, and appending each search
  report to `results/search_results.txt`.
- **`text_utils.py`** — everything about searching and formatting
  text: case-insensitive matching, collecting `(line_number, line)`
  matches, formatting a match for display, and generating the
  timestamp used in the saved report.
- **`main.py`** — ties the two modules together and runs the
  interactive "search again or exit" loop.

## Notes on error handling

- If the `documents/` folder is missing (or has no `.txt` files
  inside it), the program prints a friendly message and lets you try
  another keyword or exit — it does not crash.
- If an individual file can't be read for some reason, a warning is
  printed for that file and the search continues with the rest.
- `results/search_results.txt` and the `results/` folder itself are
  created automatically the first time they're needed, so there's
  nothing extra to set up.
- Each search is **appended** to `results/search_results.txt` (with
  its own timestamp), so the file builds up a running history of
  every search you've performed rather than overwriting the
  previous one.
