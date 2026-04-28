"""
transcript_parser.py — Parse Otter JSON exports and plain-text transcripts.

Supports:
  - Otter.ai JSON export: {"speakers": [...], "transcripts": [{"spk_id": ..., "text": ...}]}
  - Plain text with speaker labels: "Speaker A: some text"
  - Plain text without labels: treated as single-speaker monologue
"""

import json
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Turn:
    speaker: str
    text: str
    start_ms: int = 0   # millisecond offset if available, else 0


def parse(file_path: str) -> List[Turn]:
    """Return a list of Turn objects from a transcript file (.json or .txt)."""
    if file_path.endswith(".json"):
        return _parse_otter_json(file_path)
    return _parse_plain_text(file_path)


def _parse_otter_json(file_path: str) -> List[Turn]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build speaker id → name map if present
    speaker_map: dict = {}
    for spk in data.get("speakers", []):
        speaker_map[str(spk.get("id", ""))] = spk.get("name", f"Speaker {spk.get('id', '?')}")

    turns: List[Turn] = []
    for segment in data.get("transcripts", []):
        spk_id = str(segment.get("spk_id", ""))
        speaker = speaker_map.get(spk_id, f"Speaker {spk_id}")
        text = segment.get("text", "").strip()
        start_ms = segment.get("start_offset", 0)
        if text:
            turns.append(Turn(speaker=speaker, text=text, start_ms=start_ms))
    return turns


# Matches "Speaker Name: text" or "SPEAKER_01: text"
_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,40}):\s+(.+)$")


def _parse_plain_text(file_path: str) -> List[Turn]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    turns: List[Turn] = []
    current_speaker = "Speaker"
    current_lines: List[str] = []

    def flush():
        text = " ".join(current_lines).strip()
        if text:
            turns.append(Turn(speaker=current_speaker, text=text))

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        m = _LABEL_RE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_lines = [m.group(2).strip()]
        else:
            current_lines.append(line.strip())

    flush()
    return turns
