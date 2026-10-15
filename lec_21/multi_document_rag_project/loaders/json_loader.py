"""Loader for .json files. Flattens nested JSON into readable
'key: value' lines so it embeds well as text."""

import json
from typing import List, Dict, Any


def _flatten(obj: Any, prefix: str = "") -> List[str]:
    parts: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            parts.extend(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            parts.extend(_flatten(v, f"{prefix}{i}."))
    else:
        parts.append(f"{prefix.rstrip('.')}: {obj}")
    return parts


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    if isinstance(data, list):
        # Treat each top-level list item (e.g. one FAQ entry) as its own record.
        for i, item in enumerate(data):
            text = "\n".join(_flatten(item))
            if text.strip():
                records.append({"text": text, "source": file_path, "page": f"item_{i + 1}"})
    else:
        text = "\n".join(_flatten(data))
        if text.strip():
            records.append({"text": text, "source": file_path, "page": None})
    return records
