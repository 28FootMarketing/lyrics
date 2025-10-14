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
from typing import Optional, Tuple, List

# Tag readers
from mutagen import File as MutagenFile
from mutagen.id3 import USLT, SYLT, ID3
from mutagen.mp4 import MP4, MP4Tags
from mutagen.flac import FLAC

# Optional STT backends
# Try Faster-Whisper first, then OpenAI Whisper if available
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

def read_embedded_lyrics(input_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    """Return (unsynced_text, synced_list). synced_list is list of (time_seconds, line)."""
    audio = MutagenFile(input_path)
    if audio is None:
        return None, None

    # MP3 ID3
    if input_path.suffix.lower() == ".mp3":
        try:
            id3 = ID3(str(input_path))
        except Exception:
            id3 = None
        if id3:
            # Unsynced lyrics
            for frame in id3.getall("USLT"):
                if isinstance(frame, USLT) and frame.text:
                    return frame.text, None
            # Synced lyrics
            for frame in id3.getall("SYLT"):
                if isinstance(frame, SYLT) and frame.text:
                    # SYLT text is list of tuples (timestamp_ms, text) or similar
                    synced = []
                    try:
                        for t_ms, line in frame.text:
                            synced.append((t_ms/1000.0, line.strip()))
                        return None, synced
                    except Exception:
                        # Some SYLT frames store raw bytes. Fall back to unsynced if present
                        pass

    # MP4/M4A
    if input_path.suffix.lower() in {".m4a", ".mp4"}:
        try:
            mp4 = MP4(str(input_path))
            tags = mp4.tags or {}
            # Common keys: ©lyr
            for key in ["\xa9lyr", "lyrics", "LYRICS"]:
                if key in tags and tags[key]:
                    # Tags may be list-like
                    val = tags[key][0] if isinstance(tags[key], list) else tags[key]
                    return str(val), None
        except Exception:
            pass

    # FLAC
    if input_path.suffix.lower() == ".flac":
        try:
            flac = FLAC(str(input_path))
            for key in ["LYRICS", "UNSYNCEDLYRICS", "Lyrics", "lyrics"]:
                if key in flac.tags and flac.tags[key]:
                    return "\n".join(flac.tags[key]), None
        except Exception:
            pass

    # Generic Vorbis-style tags via MutagenFile
    if hasattr(audio, "tags") and audio.tags:
        for key in ["LYRICS", "UNSYNCEDLYRICS", "Lyrics", "lyrics"]:
            if key in audio.tags:
                val = audio.tags.get(key)
                if isinstance(val, list):
                    return "\n".join([str(v) for v in val]), None
                return str(val), None

    return None, None

def format_lrc_from_synced(synced: List[Tuple[float, str]]) -> str:
    lines = []
    for t, text in synced:
        m = int(t // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)  # centiseconds
        lines.append(f"[{m:02d}:{s:02d}.{cs:02d}]{text}")
    return "\n".join(lines)

def write_output(base_out: Path, unsynced: Optional[str], synced: Optional[List[Tuple[float, str]]], prefer: str) -> Path:
    """Write .txt or .lrc depending on data and preference."""
    if synced and prefer == "lrc":
        out_path = base_out.with_suffix(".lrc")
        out_path.write_text(format_lrc_from_synced(synced), encoding="utf-8")
        return out_path
    # Default to txt
    out_path = base_out.with_suffix(".txt")
    text = None
    if unsynced:
        text = unsynced
    elif synced:
        # Drop timestamps into plain text
        text = "\n".join([line for _, line in synced])
    else:
        text = ""
    out_path.write_text(text, encoding="utf-8")
    return out_path

def transcribe_with_faster_whisper(model_name: str, input_path: Path, language: Optional[str], lrc: bool) -> Tuple[str, Optional[str]]:
    """Return (.txt_content, .lrc_content or None)"""
    from faster_whisper import WhisperModel  # type: ignore
    model = WhisperModel(model_name, compute_type="int8")
    segments, info = model.transcribe(str(input_path), language=language, vad_filter=True, word_timestamps=True)
    txt_lines = []
    lrc_lines = []
    for seg in segments:
        line = seg.text.strip()
        if not line:
            continue
        txt_lines.append(line)
        if lrc:
            t = seg.start if seg.start is not None else 0.0
            m = int(t // 60)
            s = int(t % 60)
            cs = int((t - int(t)) * 100)
            lrc_lines.append(f"[{m:02d}:{s:02d}.{cs:02d}]{line}")
    return "\n".join(txt_lines), ("\n".join(lrc_lines) if lrc else None)

def transcribe_with_openai_whisper(model_name: str, input_path: Path, language: Optional[str], lrc: bool) -> Tuple[str, Optional[str]]:
    import whisper  # type: ignore
    model = whisper.load_model(model_name)
    result = model.transcribe(str(input_path), language=language, word_timestamps=True, verbose=False)
    txt_lines = [result.get("text", "").strip()]
    lrc_lines = []
    if lrc:
        # Build approximate line per segment
        for seg in result.get("segments", []):
            line = seg.get("text", "").strip()
            t = float(seg.get("start", 0.0))
            m = int(t // 60)
            s = int(t % 60)
            cs = int((t - int(t)) * 100)
            lrc_lines.append(f"[{m:02d}:{s:02d}.{cs:02d}]{line}")
    return "\n".join([l for l in txt_lines if l]), ("\n".join(lrc_lines) if lrc else None)

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
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_out_dir(Path(args.output_dir))
    base_out = out_dir / input_path.stem

    unsynced, synced = (None, None)
    if not args.force_transcribe:
        unsynced, synced = read_embedded_lyrics(input_path)

    if unsynced or synced:
        out_file = write_output(base_out, unsynced, synced, prefer=args.prefer)
        print(f"Extracted embedded lyrics -> {out_file}")
        sys.exit(0)

    # Transcription path
    fw, ow = _import_stt_backends()
    if fw is None and ow is None:
        print("No STT backend found. Install Faster-Whisper or OpenAI Whisper. See README.md", file=sys.stderr)
        sys.exit(2)

    lrc_needed = args.prefer == "lrc"
    txt_content = ""
    lrc_content = None

    try:
        if fw is not None:
            txt_content, lrc_content = transcribe_with_faster_whisper(args.model, input_path, args.language, lrc_needed)
        else:
            txt_content, lrc_content = transcribe_with_openai_whisper(args.model, input_path, args.language, lrc_needed)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        sys.exit(3)

    # Write files
    txt_path = base_out.with_suffix(".txt")
    txt_path.write_text(txt_content, encoding="utf-8")
    print(f"Transcription -> {txt_path}")
    if lrc_content:
        lrc_path = base_out.with_suffix(".lrc")
        lrc_path.write_text(lrc_content, encoding="utf-8")
        print(f"LRC -> {lrc_path}")

if __name__ == "__main__":
    main()
