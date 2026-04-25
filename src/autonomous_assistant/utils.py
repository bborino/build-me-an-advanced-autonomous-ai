from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_text(value: Any, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def extract_json_payload(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise ValueError("The model response was empty.")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    starts = [index for index, char in enumerate(candidate) if char in "{["]
    ends = [index for index, char in enumerate(candidate) if char in "}]"]
    for start in starts:
        for end in reversed(ends):
            if end <= start:
                continue
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not extract valid JSON from the model response.")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(unescape(cleaned))


def make_task_id(index: int) -> str:
    return f"T{index + 1}"


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)

