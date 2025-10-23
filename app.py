"""
Lyrics Extractor
1) Reads embedded lyrics from audio file tags when available.
2) Falls back to speech-to-text transcription using Faster-Whisper or OpenAI Whisper.
Outputs .txt by default. Can produce .lrc if timestamps exist or when transcribing with word timestamps.

Supported containers:
- MP3 (ID3: USLT unsynchronized, SYLT synchronized)
- M4A/MP4 (©lyr atom, ilst keys)
- FLAC/OGG (Vorbis comments: LYRICS, UNSYNCEDLYRICS, or custom fields)

Usage:
    python lyrics_extractor.py "path/to/song.mp3" --output-dir ./out --prefer lrc --model small

Install:
    pip install -r requirements.txt
    Ensure ffmpeg is available in PATH for transcription.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List, Iterable, Dict

# Tag readers
from mutagen import File as MutagenFile
from mutagen.id3 import USLT, SYLT, ID3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
try:
    from mutagen.oggvorbis import OggVorbis
except Exception:
    OggVorbis = None  # optional

# --------------------------
# Optional STT backends
# --------------------------
def _import_stt_backends():
    fw = None
    ow = None
    try:
        from faster_whisper import WhisperModel  # type: ignore
        fw = WhisperModel
    except Exception:
        pass
    try:
        import whisper  # type: ignore
        ow = whisper
    except Exception:
        pass
    return fw, ow

# --------------------------
# Tag extraction helpers
# --------------------------
def _first_nonempty(values: Iterable[Optional[str]]) -> Optional[str]:
    for v in values:
        if v:
            s = str(v).strip()
            if s:
                return s
    return None

def _read_id3_lyrics(input_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        id3 = ID3(str(input_path))
    except Exception:
        return None, None

    # Prefer unsynced USLT first
    unsynced_candidates = []
    for frame in id3.getall("USLT"):
        if isinstance(frame, USLT) and frame.text:
            # Collect by length to choose the most complete if multiple frames exist
            unsynced_candidates.append(str(frame.text).strip())
    unsynced = max(unsynced_candidates, key=len) if unsynced_candidates else None

    # Then check synchronized SYLT
    synced: List[Tuple[float, str]] = []
    for frame in id3.getall("SYLT"):
        if not isinstance(frame, SYLT) or not frame.text:
            continue
        # Mutagen SYLT.text can be a list of tuples [(timestamp_ms, text)] or [(text, timestamp_ms)]
        # Try both orders defensively
        local_synced: List[Tuple[float, str]] = []
        try:
            for item in frame.text:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    a, b = item[0], item[1]
                    if isinstance(a, int) and not isinstance(b, int):
                        t_ms, line = a, str(b)
                    elif isinstance(b, int) and not isinstance(a, int):
                        t_ms, line = b, str(a)
                    else:
                        # Fallback skip if unreadable
                        continue
                    local_synced.append((max(t_ms, 0) / 1000.0, line.strip()))
        except Exception:
            local_synced = []
        if local_synced:
            synced.extend(local_synced)

    return unsynced if unsynced else None, (synced if synced else None)

def _read_mp4_lyrics(input_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        mp4 = MP4(str(input_path))
        tags = mp4.tags or {}
    except Exception:
        return None, None

    # Normalize MP4 atoms to strings; common: '\xa9lyr'
    keys = ["\xa9lyr", "lyrics", "LYRICS"]
    for k in keys:
        if k in tags and tags[k]:
            val = tags[k]
            if isinstance(val, list):
                val = val[0]
            return str(val), None
    return None, None

def _read_flac_lyrics(input_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        flac = FLAC(str(input_path))
    except Exception:
        return None, None
    # Vorbis comments are case-insensitive; use get() to avoid KeyError
    for key in ("LYRICS", "UNSYNCEDLYRICS", "Lyrics", "lyrics"):
        vals = flac.tags.get(key, [])
        if vals:
            return "\n".join([str(v) for v in vals]), None
    return None, None

def _read_vorbis_like(audio) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    tags = getattr(audio, "tags", None)
    if not tags:
        return None, None
    # Normalize keys to upper for lookup
    upper_map: Dict[str, List[str]] = {}
    try:
        for k in tags.keys():
            v = tags.get(k)
            if v is None:
                continue
            vals = v if isinstance(v, list) else [v]
            upper_map.setdefault(str(k).upper(), [])
            upper_map[str(k).upper()].extend([str(x) for x in vals])
    except Exception:
        return None, None

    for key in ("LYRICS", "UNSYNCEDLYRICS"):
        vals = upper_map.get(key, [])
        if vals:
            return "\n".join(vals), None
    return None, None

def read_embedded_lyrics(input_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    """Return (unsynced_text, synced_list). synced_list is [(time_seconds, line)]."""
    audio = MutagenFile(input_path)
    if audio is None:
        return None, None

    suf = input_path.suffix.lower()
    if suf == ".mp3":
        return _read_id3_lyrics(input_path)
    if suf in {".m4a", ".mp4"}:
        return _read_mp4_lyrics(input_path)
    if suf == ".flac":
        return _read_flac_lyrics(input_path)
    if suf == ".ogg" and OggVorbis is not None:
        # OGG Vorbis typically uses Vorbis comments
        return _read_vorbis_like(audio)

    # Generic Vorbis-style tags if present
    return _read_vorbis_like(audio)

# --------------------------
# Formatting
# --------------------------
def _format_ts(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"[{m:02d}:{s:02d}.{cs:02d}]"

def format_lrc_from_synced(
    synced: List[Tuple[float, str]],
    title: Optional[str] = None,
    artist: Optional[str] = None,
) -> str:
    lines: List[str] = []
    if artist:
        lines.append(f"[ar:{artist}]")
    if title:
        lines.append(f"[ti:{title}]")
    for t, text in synced:
        if text:
            lines.append(f"{_format_ts(max(t, 0.0))}{text.strip()}")
    return "\n".join(lines)

def write_output(
    base_out: Path,
    unsynced: Optional[str],
    synced: Optional[List[Tuple[float, str]]],
    prefer: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
) -> Path:
    """Write .txt or .lrc depending on data and preference."""
    if synced and prefer == "lrc":
        out_path = base_out.with_suffix(".lrc")
        out_path.write_text(format_lrc_from_synced(synced, title, artist), encoding="utf-8")
        return out_path

    out_path = base_out.with_suffix(".txt")
    text = unsynced or ("\n".join([line for _, line in synced]) if synced else "")
    out_path.write_text(text, encoding="utf-8")
    return out_path

# --------------------------
# STT backends
# --------------------------
def transcribe_with_faster_whisper(
    model_name: str,
    input_path: Path,
    language: Optional[str],
    lrc: bool,
    compute_type: str,
) -> Tuple[str, Optional[str]]:
    from faster_whisper import WhisperModel  # type: ignore
    # compute_type examples: "auto", "int8", "int8_float16", "float16"
    model = WhisperModel(model_name, compute_type=compute_type or "auto")
    segments, info = model.transcribe(
        str(input_path),
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    txt_lines: List[str] = []
    lrc_lines: List[str] = []
    for seg in segments:
        line = (seg.text or "").strip()
        if not line:
            continue
        txt_lines.append(line)
        if lrc:
            t = float(seg.start or 0.0)
            lrc_lines.append(f"{_format_ts(t)}{line}")
    return "\n".join(txt_lines), ("\n".join(lrc_lines) if lrc else None)

def transcribe_with_openai_whisper(
    model_name: str,
    input_path: Path,
    language: Optional[str],
    lrc: bool,
) -> Tuple[str, Optional[str]]:
    import whisper  # type: ignore
    model = whisper.load_model(model_name)
    result = model.transcribe(str(input_path), language=language, word_timestamps=True, verbose=False)
    txt = (result.get("text") or "").strip()
    if not lrc:
        return txt, None
    lrc_lines: List[str] = []
    for seg in result.get("segments", []):
        line = (seg.get("text") or "").strip()
        t = float(seg.get("start", 0.0))
        if line:
            lrc_lines.append(f"{_format_ts(t)}{line}")
    return txt, ("\n".join(lrc_lines) if lrc_lines else None)

# --------------------------
# CLI
# --------------------------
def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def main():
    ap = argparse.ArgumentParser(description="Extract embedded lyrics or transcribe audio to lyrics.")
    ap.add_argument("input", help="Path to audio file")
    ap.add_argument("--output-dir", default="./lyrics_out", help="Directory for outputs")
    ap.add_argument("--prefer", choices=["txt", "lrc"], default="txt", help="Preferred output when synced data is available or during transcription")
    ap.add_argument("--force-transcribe", action="store_true", help="Skip tag extraction and transcribe directly")
    ap.add_argument("--model", default="small", help="Whisper model name. Examples: tiny, base, small, medium, large-v2, or CTranslate2 names like 'small'")
    ap.add_argument("--language", default=None, help="Language code. Example: en")
    ap.add_argument("--backend", choices=["auto", "faster-whisper", "openai-whisper"], default="auto", help="Choose STT backend when both are installed")
    ap.add_argument("--compute-type", default="auto", help="Faster-Whisper compute type. Example: auto, int8, int8_float16, float16")
    ap.add_argument("--title", default=None, help="Optional title for LRC headers")
    ap.add_argument("--artist", default=None, help="Optional artist for LRC headers")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_out_dir(Path(args.output_dir))
    base_out = out_dir / input_path.stem

    # Try tags first unless forced
    unsynced, synced = (None, None)
    if not args.force_transcribe:
        if input_path.suffix.lower() == ".mp3":
            unsynced, synced = _read_id3_lyrics(input_path)
        elif input_path.suffix.lower() in {".m4a", ".mp4"}:
            unsynced, synced = _read_mp4_lyrics(input_path)
        elif input_path.suffix.lower() == ".flac":
            unsynced, synced = _read_flac_lyrics(input_path)
        else:
            audio = MutagenFile(input_path)
            if audio is not None:
                unsynced, synced = _read_vorbis_like(audio)

    if unsynced or synced:
        out_file = write_output(base_out, unsynced, synced, prefer=args.prefer, title=args.title, artist=args.artist)
        print(f"Extracted embedded lyrics -> {out_file}")
        sys.exit(0)

    # Transcription path
    fw, ow = _import_stt_backends()
    backend = args.backend
    if backend == "auto":
        backend = "faster-whisper" if fw is not None else ("openai-whisper" if ow is not None else "none")

    if backend == "none":
        print("No STT backend found. Install Faster-Whisper or OpenAI Whisper. See README.md", file=sys.stderr)
        sys.exit(2)

    lrc_needed = args.prefer == "lrc"
    try:
        if backend == "faster-whisper":
            if fw is None:
                raise RuntimeError("Faster-Whisper not installed")
            txt_content, lrc_content = transcribe_with_faster_whisper(args.model, input_path, args.language, lrc_needed, args.compute_type)
        else:
            if ow is None:
                raise RuntimeError("OpenAI Whisper not installed")
            txt_content, lrc_content = transcribe_with_openai_whisper(args.model, input_path, args.language, lrc_needed)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        sys.exit(3)

    # Write files
    txt_path = base_out.with_suffix(".txt")
    txt_path.write_text(txt_content or "", encoding="utf-8")
    print(f"Transcription -> {txt_path}")
    if lrc_content:
        lrc_path = base_out.with_suffix(".lrc")
        lrc_path.write_text(lrc_content, encoding="utf-8")
        print(f"LRC -> {lrc_path}")

if __name__ == "__main__":
    main()
