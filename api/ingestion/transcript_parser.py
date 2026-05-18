"""
transcript_parser.py — Parse Otter JSON exports and plain-text transcripts.

Supports:
  - Otter.ai JSON export: {"speakers": [...], "transcripts": [{"spk_id": ..., "text": ...}]}
  - WebVTT (.vtt): WEBVTT header + timestamp blocks
  - SRT subtitles: numbered blocks with HH:MM:SS,mmm --> HH:MM:SS,mmm timestamps
  - Zoom chat/transcript: "HH:MM:SS" timestamp on a line then "Speaker: text"
  - MS Teams: "Speaker Name\nHH:MM AM/PM\ntext" blocks
  - Plain text with speaker labels: "Speaker Name: some text"
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


# ── Format detection ─────────────────────────────────────────────────────────

_VTT_HEADER_RE  = re.compile(r"^WEBVTT", re.IGNORECASE)
_SRT_BLOCK_RE   = re.compile(r"^\d+\s*$")
_SRT_TS_RE      = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
_VTT_TS_RE      = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}")
# Zoom standalone timestamp line: "00:01:23" or "0:01:23"
_ZOOM_TS_LINE   = re.compile(r"^\d{1,2}:\d{2}:\d{2}\s*$")
# Teams block: speaker name line followed by time like "10:05 AM"
_TEAMS_TIME_RE  = re.compile(r"^\d{1,2}:\d{2}\s*[AaPp][Mm]\s*$")


def _detect_format(lines: List[str]) -> str:
    """Return one of: 'vtt' | 'srt' | 'zoom' | 'teams' | 'plain'."""
    head = lines[:20]
    if head and _VTT_HEADER_RE.match(head[0].strip()):
        return "vtt"
    # SRT: look for a numbered line followed by a timestamp line in the first 40 lines
    for i, line in enumerate(lines[:40]):
        if _SRT_BLOCK_RE.match(line.strip()) and i + 1 < len(lines):
            if _SRT_TS_RE.search(lines[i + 1]):
                return "srt"
    # Zoom: standalone timestamp-only lines like "00:01:23"
    ts_only_count = sum(1 for l in head if _ZOOM_TS_LINE.match(l.strip()))
    if ts_only_count >= 2:
        return "zoom"
    # Teams: "HH:MM AM" time-only lines
    teams_time_count = sum(1 for l in head if _TEAMS_TIME_RE.match(l.strip()))
    if teams_time_count >= 2:
        return "teams"
    return "plain"


# ── Format-specific normalizers → canonical "Speaker: text" lines ────────────

def _normalize_vtt(lines: List[str]) -> List[str]:
    """Convert WebVTT to canonical Speaker: text lines.

    VTT blocks look like:
      00:00:05.000 --> 00:00:10.000
      Speaker Name: text here
    or just:
      00:00:05.000 --> 00:00:10.000
      text here (no speaker label — kept as-is for the plain parser)
    """
    out = []
    skip_next = False
    for line in lines:
        stripped = line.strip()
        if not stripped or _VTT_HEADER_RE.match(stripped):
            continue
        if _VTT_TS_RE.search(stripped):
            # The next non-empty line(s) are content; skip the timestamp itself
            continue
        # Strip VTT cue settings: lines where EVERY space-separated token contains
        # a colon (e.g. "align:middle position:50% line:0%").
        # The old heuristic (starts with lowercase + no colon in first word) incorrectly
        # dropped real content lines like "this was discussed" or "looks good on our end".
        tokens = stripped.split()
        if tokens and all(":" in t for t in tokens):
            continue
        out.append(stripped)
    return out


def _normalize_srt(lines: List[str]) -> List[str]:
    """Convert SRT subtitles to plain text lines, stripping index numbers and timestamps."""
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _SRT_BLOCK_RE.match(stripped):
            continue
        if _SRT_TS_RE.search(stripped):
            continue
        # Strip HTML tags common in SRT (<i>, <b>, <font>)
        stripped = re.sub(r"<[^>]+>", "", stripped).strip()
        if stripped:
            out.append(stripped)
    return out


def _normalize_zoom(lines: List[str]) -> List[str]:
    """Convert Zoom transcript format to canonical Speaker: text lines.

    Zoom plain-text exports look like:
      00:01:23
      Alice Johnson
      Some words she said.

      00:01:45
      Bob Smith
      His reply here.
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if _ZOOM_TS_LINE.match(line):
            # Next non-empty line is the speaker name
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                speaker = lines[i].strip()
                i += 1
                # Collect text lines until next timestamp or blank
                text_parts = []
                while i < len(lines) and not _ZOOM_TS_LINE.match(lines[i].strip()):
                    t = lines[i].strip()
                    if t:
                        text_parts.append(t)
                    elif text_parts:
                        break
                    i += 1
                if speaker and text_parts:
                    out.append(f"{speaker}: {' '.join(text_parts)}")
        else:
            if line:
                out.append(line)
            i += 1
    return out


def _normalize_teams(lines: List[str]) -> List[str]:
    """Convert MS Teams transcript format to canonical Speaker: text lines.

    Teams exports typically look like:
      Alice Johnson
      10:05 AM
      Text of what Alice said.

      Bob Smith
      10:06 AM
      Bob's reply.
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Check if next non-empty line is a Teams time stamp
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and _TEAMS_TIME_RE.match(lines[j].strip()):
            speaker = line
            # Skip the timestamp line
            k = j + 1
            text_parts = []
            while k < len(lines):
                t = lines[k].strip()
                if not t:
                    break
                # Stop if we hit another speaker block (next speaker + time pattern)
                m = k + 1
                while m < len(lines) and not lines[m].strip():
                    m += 1
                if m < len(lines) and _TEAMS_TIME_RE.match(lines[m].strip()):
                    text_parts.append(t)
                    break
                text_parts.append(t)
                k += 1
            if speaker and text_parts:
                out.append(f"{speaker}: {' '.join(text_parts)}")
            i = k + 1
        else:
            if line:
                out.append(line)
            i += 1
    return out


# ── Main plain-text parser ────────────────────────────────────────────────────

# Matches "Speaker Name: text" or "SPEAKER_01: text".
# À-ɏ covers Latin Extended A+B (é, ü, ñ, ø, etc.) so that
# international names like "María García" or "André" are matched correctly.
_LABEL_RE = re.compile(
    r"^([A-Za-zÀ-ɏ][A-Za-z0-9À-ɏ _-]{0,40}):\s+(.+)$"
)

# Metadata keys commonly found in transcript header blocks — these are not speakers.
_HEADER_KEYS = frozenset({
    "date", "duration", "attendees", "participants", "location", "meeting",
    "subject", "time", "organizer", "host", "agenda", "title", "summary",
    "project", "call", "recorded", "transcribed",
})


def _strip_transcript_header(lines: List[str]) -> List[str]:
    """Remove meeting metadata header lines before the dialogue starts.

    Strategy: skip everything before the first '---' separator line. If no
    separator is found, scan from the top and skip any line whose colon-prefix
    key matches a known metadata keyword (Date, Attendees, Duration, etc.).
    This prevents those keys from being parsed as speaker names.
    """
    # Fast path: explicit --- separator
    for i, line in enumerate(lines):
        if line.strip() in ("---", "—", "***", "==="):
            return lines[i + 1:]

    # No separator — strip leading metadata-key lines only
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _LABEL_RE.match(stripped)
        if m and m.group(1).strip().lower() in _HEADER_KEYS:
            start = i + 1  # skip this metadata line, continue scanning
        else:
            break  # first non-metadata line — dialogue starts here
    return lines[start:]


def _parse_plain_text(file_path: str) -> List[Turn]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    lines = [l.rstrip() for l in raw_lines]

    # Detect and normalise third-party formats before the label parser runs
    fmt = _detect_format(lines)
    if fmt == "vtt":
        lines = _normalize_vtt(lines)
    elif fmt == "srt":
        lines = _normalize_srt(lines)
    elif fmt == "zoom":
        lines = _normalize_zoom(lines)
    elif fmt == "teams":
        lines = _normalize_teams(lines)
    else:
        # Plain format: strip meeting metadata header before label parsing
        lines = _strip_transcript_header(lines)
    # "plain" falls through to label parser below

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
