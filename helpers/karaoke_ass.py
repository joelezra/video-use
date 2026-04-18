"""
karaoke_ass.py — build an ASS subtitle file with word-level karaoke highlighting.

Mirrors the behavior of build_master_srt() in render.py (same chunking, same
output-timeline offset handling) but emits ASS with per-word events so only
the word currently being spoken is highlighted; the rest of the chunk stays
baseline white. This is the "viral shorts" Submagic-style look.

Strategy (simple, battle-tested):
  - Group words into ≤2-word chunks, break on punctuation (matches SRT).
  - For each chunk, emit N Dialogue events (one per word in the chunk),
    each active during that word's speaking window. Each event shows the
    FULL chunk text with only the active word color-overridden.
  - When the chunk ends, all events expire → caption disappears until the
    next chunk fires.

Output: a single .ass file referencing our vertical-Shorts resolution
(PlayResX=1080, PlayResY=1920) so coordinates match the rendered frame
and we don't depend on libass's implicit scaling heuristics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PUNCT_BREAK = ".!?,;:"


def _ass_time(seconds: float) -> str:
    """ASS time format: H:MM:SS.CC (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    cs_total = int(round(seconds * 100))
    h, rem = divmod(cs_total, 3600 * 100)
    m, rem = divmod(rem, 60 * 100)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters that would confuse ASS Dialogue Text parsing."""
    # Backslash and braces are special in ASS. Newlines become \N (literal).
    return (
        text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
    )


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def _chunk_words(words: list[dict], max_words: int = 2) -> list[list[dict]]:
    """2-word chunks, break on any trailing punctuation (matches SRT builder)."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        current.append(w)
        ends_punct = text[-1] in PUNCT_BREAK
        if len(current) >= max_words or ends_punct:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


# ASS colors are &HAABBGGRR (alpha-blue-green-red). 00 alpha = opaque.
# Defaults tuned for vertical Shorts on busy backgrounds:
#   baseline = white with black outline
#   active   = bright yellow with black outline, slight scale-up
_DEFAULT_ASS_STYLE = {
    "Name": "Karaoke",
    "Fontname": "Helvetica",
    "Fontsize": "84",
    "PrimaryColour": "&H00FFFFFF",   # white (baseline)
    "SecondaryColour": "&H0000F0FF", # bright yellow (for \k karaoke-transition)
    "OutlineColour": "&H00000000",   # black outline
    "BackColour": "&H80000000",      # translucent black shadow
    "Bold": "-1",
    "Italic": "0",
    "Underline": "0",
    "StrikeOut": "0",
    "ScaleX": "100",
    "ScaleY": "100",
    "Spacing": "0",
    "Angle": "0",
    "BorderStyle": "1",  # outline + shadow
    "Outline": "5",
    "Shadow": "2",
    "Alignment": "8",    # top-center (above source's burned-in bottom subs)
    "MarginL": "60",
    "MarginR": "60",
    "MarginV": "240",    # distance from top
    "Encoding": "1",
}


_STYLE_FIELDS = [
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
    "OutlineColour", "BackColour", "Bold", "Italic", "Underline", "StrikeOut",
    "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline", "Shadow",
    "Alignment", "MarginL", "MarginR", "MarginV", "Encoding",
]


def _style_line(style: dict) -> str:
    return "Style: " + ",".join(str(style.get(f, "")) for f in _STYLE_FIELDS)


def _format_chunk_event(
    chunk: list[dict],
    active_idx: int,
    seg_start: float,
    seg_offset: float,
    next_word_start_abs: float | None,
    chunk_end_abs: float,
    active_color: str = "&H0000F0FF",   # yellow (BGR)
    active_scale: int = 115,            # percent
) -> tuple[float, float, str]:
    """Build a single Dialogue event where only `active_idx` word is highlighted.

    Each word's event runs from its own start until the next word's start
    (or the chunk end for the last word), so the chunk stays continuously
    on-screen while the highlight slides word-to-word. Eliminates flicker
    between word boundaries.
    """
    word = chunk[active_idx]
    w_start_abs = float(word.get("start", seg_start))

    # Extend until next word begins, or chunk end if this is the last word
    if next_word_start_abs is not None:
        end_abs = next_word_start_abs
    else:
        end_abs = chunk_end_abs

    out_start = max(0.0, w_start_abs - seg_start) + seg_offset
    out_end = max(out_start + 0.05, end_abs - seg_start + seg_offset)

    parts: list[str] = []
    for i, w in enumerate(chunk):
        text = (w.get("text") or "").strip()
        if not text:
            continue
        token = text.rstrip(",;:").upper()
        token = _escape_ass_text(token)
        if i == active_idx:
            parts.append(
                f"{{\\1c{active_color}\\fscx{active_scale}\\fscy{active_scale}}}"
                f"{token}"
                f"{{\\1c&H00FFFFFF&\\fscx100\\fscy100}}"
            )
        else:
            parts.append(token)
    line = " ".join(parts)
    return out_start, out_end, line


def build_master_ass(
    edl: dict,
    edit_dir: Path,
    out_path: Path,
    *,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    style_overrides: dict | None = None,
    active_color: str = "&H0000F0FF",
    active_scale: int = 115,
    max_words_per_chunk: int = 2,
) -> None:
    """Build an output-timeline karaoke-style ASS from per-source transcripts.

    Same chunking / offset logic as build_master_srt in render.py, but emits
    per-word Dialogue events with color+scale override on the active word.
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    style = dict(_DEFAULT_ASS_STYLE)
    if style_overrides:
        style.update(style_overrides)

    events: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping karaoke for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text())
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)
        chunks = _chunk_words(words_in_seg, max_words=max_words_per_chunk)

        for chunk in chunks:
            chunk_end_abs = float(chunk[-1].get("end", seg_end))
            for idx in range(len(chunk)):
                # Look up next word's start for seamless chunk-level display
                if idx + 1 < len(chunk):
                    next_start = float(chunk[idx + 1].get("start", chunk_end_abs))
                else:
                    next_start = None
                out_start, out_end, text_line = _format_chunk_event(
                    chunk, idx, seg_start, seg_offset,
                    next_word_start_abs=next_start,
                    chunk_end_abs=chunk_end_abs,
                    active_color=active_color, active_scale=active_scale,
                )
                if out_end <= out_start:
                    continue
                events.append((out_start, out_end, text_line))

        seg_offset += seg_duration

    events.sort(key=lambda e: e[0])

    # Compose ASS file
    lines: list[str] = [
        "[Script Info]",
        "; Generated by content-engine karaoke_ass.py",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 2",           # smart wrap, no line break on spaces
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: " + ", ".join(_STYLE_FIELDS),
        _style_line(style),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    style_name = style.get("Name", "Karaoke")
    for a, b, t in events:
        lines.append(
            f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},{style_name},,0,0,0,,{t}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"master ASS → {out_path.name} ({len(events)} events, karaoke style)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build karaoke ASS from an EDL v1 + transcripts")
    ap.add_argument("edl_path", help="edl_v1.json with sources + ranges")
    ap.add_argument("--edit-dir", required=True, help="dir containing transcripts/")
    ap.add_argument("--out", required=True, help="output .ass path")
    ap.add_argument("--active-color", default="&H0000F0FF", help="BGR color for active word (default yellow)")
    ap.add_argument("--active-scale", type=int, default=115, help="scale-up percent for active word")
    args = ap.parse_args()

    edl = json.loads(Path(args.edl_path).read_text())
    build_master_ass(
        edl, Path(args.edit_dir), Path(args.out),
        active_color=args.active_color,
        active_scale=args.active_scale,
    )
