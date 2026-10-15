"""Loader for .csv files. Each row becomes its own record so retrieval
can surface individual rows (e.g. one HR policy line) as a source."""

from typing import List, Dict

import pandas as pd


def load_csv(file_path: str) -> List[Dict]:
    df = pd.read_csv(file_path)
    records = []
    for i, row in df.iterrows():
        row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
        if row_text.strip():
            records.append({"text": row_text, "source": file_path, "page": f"row_{i + 1}"})
    return records
