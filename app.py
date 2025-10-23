"""
Lyrics Extractor App
1) Reads embedded lyrics from audio file tags when available.
2) Falls back to speech-to-text transcription using Faster-Whisper or OpenAI Whisper.
Outputs .txt by default. Can produce .lrc if timestamps exist or during transcription.

Supported containers:
- MP3 (ID3: USLT unsynchronized, SYLT synchronized)
- M4A/MP4 (©lyr atom, ilst keys)
- FLAC/OGG (Vorbis comments: LYRICS, UNSYNCEDLYRICS, or custom fields)
- WAV and AAC are accepted for transcription only

Usage:
    streamlit run streamlit_app.py

Prerequisites:
    pip install -r requirements.txt
    Ensure ffmpeg is available on PATH for transcription.

Notes:
    Use Faster-Whisper on CPU with compute_type=int8 for speed.
    Use large-v2 or medium on modern GPUs for quality.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import io
import os
import shutil
import tempfile
import subprocess
import sys
import time

import streamlit as st
from mutagen import File as MutagenFile
from mutagen.id3 import USLT, SYLT, ID3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC

try:
    from mutagen.oggvorbis import OggVorbis  # noqa: F401
except Exception:
    OggVorbis = None

# --------------------------
# App constants and helpers
# --------------------------
APP_TITLE = "Lyrics Extractor"
MAX_FILE_MB = 200  # safety guard
ALLOWED_UPLOADS = ["mp3", "m4a", "mp4", "flac", "ogg", "wav", "aac"]
DEFAULT_MODEL = "small"
DEFAULT_BACKEND = "Auto"
DEFAULT_PREFER = "txt"
DEFAULT_COMPUTE = "auto"

def _format_ts(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"[{m:02d}:{s:02d}.{cs:02d}]"

def _read_id3_lyrics(tmp_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        id3 = ID3(str(tmp_path))
    except Exception:
        return None, None

    unsynced = None
    uns = []
    for frame in id3.getall("USLT"):
        if isinstance(frame, USLT) and frame.text:
            uns.append(str(frame.text).strip())
    if uns:
        unsynced = max(uns, key=len)

    synced: List[Tuple[float, str]] = []
    for frame in id3.getall("SYLT"):
        if not isinstance(frame, SYLT) or not frame.text:
            continue
        local = []
        try:
            for item in frame.text:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    a, b = item[0], item[1]
                    if isinstance(a, int) and not isinstance(b, int):
                        t_ms, line = a, str(b)
                    elif isinstance(b, int) and not isinstance(a, int):
                        t_ms, line = b, str(a)
                    else:
                        continue
                    local.append((max(t_ms, 0) / 1000.0, line.strip()))
        except Exception:
            local = []
        if local:
            synced.extend(local)

    return (unsynced or None), (synced or None)

def _read_mp4_lyrics(tmp_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        mp4 = MP4(str(tmp_path))
        tags = mp4.tags or {}
    except Exception:
        return None, None
    for key in ["\xa9lyr", "lyrics", "LYRICS"]:
        if key in tags and tags[key]:
            val = tags[key]
            if isinstance(val, list):
                val = val[0]
            return str(val), None
    return None, None

def _read_flac_lyrics(tmp_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    try:
        flac = FLAC(str(tmp_path))
    except Exception:
        return None, None
    for key in ("LYRICS", "UNSYNCEDLYRICS", "Lyrics", "lyrics"):
        vals = flac.tags.get(key, [])
        if vals:
            return "\n".join([str(v) for v in vals]), None
    return None, None

def _read_vorbis_like(audio) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    tags = getattr(audio, "tags", None)
    if not tags:
        return None, None
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

def read_embedded_lyrics(tmp_path: Path) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]]]:
    audio = MutagenFile(tmp_path)
    if audio is None:
        return None, None
    suf = tmp_path.suffix.lower()
    if suf == ".mp3":
        return _read_id3_lyrics(tmp_path)
    if suf in {".m4a", ".mp4"}:
        return _read_mp4_lyrics(tmp_path)
    if suf == ".flac":
        return _read_flac_lyrics(tmp_path)
    return _read_vorbis_like(audio)

def format_lrc_from_synced(
    synced: List[Tuple[float, str]],
    title: Optional[str],
    artist: Optional[str],
) -> str:
    lines = []
    if artist:
        lines.append(f"[ar:{artist}]")
    if title:
        lines.append(f"[ti:{title}]")
    for t, text in synced:
        if text:
            lines.append(f"{_format_ts(max(t, 0.0))}{text.strip()}")
    return "\n".join(lines)

# --------------------------
# System checks and utilities
# --------------------------
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def convert_to_wav(bytes_in: bytes, tmp_out: Path) -> Path:
    with tempfile.NamedTemporaryFile(delete=False) as f_in:
        f_in.write(bytes_in)
        in_path = Path(f_in.name)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(tmp_out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        in_path.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:2000])
    return tmp_out

@st.cache_resource(show_spinner=False)
def _cache_fw_model(model_name: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, compute_type=compute_type or "auto")

@st.cache_resource(show_spinner=False)
def _cache_ow_model(model_name: str):
    import whisper
    return whisper.load_model(model_name)

def available_backends() -> Tuple[bool, bool]:
    fw_available, ow_available = False, False
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        fw_available = True
    except Exception:
        pass
    try:
        import whisper  # noqa: F401
        ow_available = True
    except Exception:
        pass
    return fw_available, ow_available

# --------------------------
# Transcription
# --------------------------
def transcribe_faster_whisper(
    model_name: str,
    wav_path: Path,
    language: Optional[str],
    want_lrc: bool,
    compute_type: str,
    word_level_lrc: bool,
) -> Tuple[str, Optional[str], Dict[str, str]]:
    model = _cache_fw_model(model_name, compute_type)
    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    txt_lines, lrc_lines = [], []
    meta = {
        "language": getattr(info, "language", "") or "",
        "duration_sec": f"{getattr(info, 'duration', 0.0):.2f}",
    }

    # Build transcript and optional LRC
    for seg in segments:
        line = (seg.text or "").strip()
        if not line:
            continue
        txt_lines.append(line)
        if want_lrc:
            if word_level_lrc and getattr(seg, "words", None):
                for w in seg.words:
                    if getattr(w, "start", None) is None:
                        continue
                    t = float(w.start)
                    token = (w.word or "").strip()
                    if token:
                        lrc_lines.append(f"{_format_ts(t)}{token}")
            else:
                t = float(seg.start or 0.0)
                lrc_lines.append(f"{_format_ts(t)}{line}")

    return "\n".join(txt_lines), ("\n".join(lrc_lines) if want_lrc else None), meta

def transcribe_openai_whisper(
    model_name: str,
    wav_path: Path,
    language: Optional[str],
    want_lrc: bool,
) -> Tuple[str, Optional[str], Dict[str, str]]:
    model = _cache_ow_model(model_name)
    result = model.transcribe(str(wav_path), language=language, word_timestamps=True, verbose=False)
    txt = (result.get("text") or "").strip()
    meta = {
        "language": str(result.get("language", "") or ""),
        "duration_sec": f"{float(result.get('duration', 0.0)):.2f}" if "duration" in result else "",
    }
    if not want_lrc:
        return txt, None, meta
    lrc_lines: List[str] = []
    for seg in result.get("segments", []):
        line = (seg.get("text") or "").strip()
        t = float(seg.get("start", 0.0))
        if line:
            lrc_lines.append(f"{_format_ts(t)}{line}")
    return txt, ("\n".join(lrc_lines) if lrc_lines else None), meta

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎵")
st.title(APP_TITLE)
st.caption("Upload an audio file. The app will pull embedded lyrics if present. If not, it can transcribe.")

with st.sidebar:
    st.subheader("Settings")
    backend = st.selectbox("Backend", [DEFAULT_BACKEND, "Faster-Whisper", "OpenAI Whisper"])
    model_name = st.text_input("Model", value=DEFAULT_MODEL, help="Examples: tiny, base, small, medium, large-v2")
    language = st.text_input("Language code", value="", help="Example: en, es, fr. Leave blank for auto-detect.")
    prefer_format = st.selectbox("Preferred Output", [DEFAULT_PREFER, "lrc"])
    compute_type = st.text_input("Faster-Whisper compute_type", value=DEFAULT_COMPUTE, help="auto, int8, int8_float16, float16")
    word_level = st.checkbox("Word-level LRC (Faster-Whisper only)", value=False)
    title_tag = st.text_input("LRC Title (optional)", value="")
    artist_tag = st.text_input("LRC Artist (optional)", value="")
    force_transcribe = st.checkbox("Force transcription", value=False)
    st.divider()
    st.caption("Tip: Use compute_type=int8 on CPU for speed. Use float16 on GPUs for best balance.")

uploaded = st.file_uploader(
    "Choose an audio file",
    type=ALLOWED_UPLOADS,
    help="MP3, M4A, MP4, FLAC, OGG, WAV, or AAC",
)

run = st.button("Process")

if run:
    if not uploaded:
        st.error("Please upload an audio file first.")
        st.stop()

    # Size guardrail
    file_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if file_mb > MAX_FILE_MB:
        st.error(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller file.")
        st.stop()

    # ffmpeg presence for non-tag path
    if not ffmpeg_available():
        st.warning("ffmpeg not detected on PATH. Tag extraction will work, but transcription will fail until ffmpeg is installed.")

    # Unique temp workspace
    suffix = Path(uploaded.name).suffix.lower() or ".bin"
    tmp_dir = Path(tempfile.mkdtemp(prefix="lyrics_"))
    tmp_path = tmp_dir / f"upload{suffix}"
    wav_tmp = tmp_dir / "upload_16k.wav"

    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        # Step 1. Try embedded tags unless forced
        unsynced, synced = (None, None)
        if not force_transcribe:
            with st.status("Reading embedded tags...", expanded=False):
                unsynced, synced = read_embedded_lyrics(tmp_path)

        # Step 2. If found, output
        if (unsynced or synced) and not force_transcribe:
            st.success("Embedded lyrics found.")
            base_name = Path(uploaded.name).stem
            if prefer_format == "lrc" and synced:
                lrc_text = format_lrc_from_synced(synced, title_tag or None, artist_tag or None)
                st.code(lrc_text, language="text")
                st.download_button("Download .lrc", data=lrc_text, file_name=f"{base_name}.lrc", mime="text/plain")
            else:
                txt_text = unsynced if unsynced else "\n".join([line for _, line in synced])
                st.code(txt_text, language="text")
                st.download_button("Download .txt", data=txt_text, file_name=f"{base_name}.txt", mime="text/plain")
            st.stop()

        # Step 3. Transcription path
        st.info("No embedded lyrics found or transcription forced. Starting transcription...")

        try:
            wav_tmp = convert_to_wav(tmp_path.read_bytes(), wav_tmp)
        except Exception as e:
            st.error("ffmpeg conversion failed. Ensure ffmpeg is installed.")
            with st.expander("Details"):
                st.text(str(e))
            st.stop()

        want_lrc = prefer_format == "lrc"
        lang = language or None

        fw_available, ow_available = available_backends()
        chosen = backend
        if chosen == "Auto":
            chosen = "Faster-Whisper" if fw_available else ("OpenAI Whisper" if ow_available else "None")

        if chosen == "None" or (chosen == "Faster-Whisper" and not fw_available) or (chosen == "OpenAI Whisper" and not ow_available):
            st.error("No valid STT backend available. Install Faster-Whisper or OpenAI Whisper.")
            st.stop()

        start = time.time()
        try:
            with st.status("Transcribing...", expanded=False):
                if chosen == "Faster-Whisper":
                    txt_text, lrc_text, meta = transcribe_faster_whisper(
                        model_name, wav_tmp, lang, want_lrc, compute_type, word_level_lrc=word_level
                    )
                else:
                    txt_text, lrc_text, meta = transcribe_openai_whisper(
                        model_name, wav_tmp, lang, want_lrc
                    )
        except Exception as e:
            st.error("Transcription failed.")
            with st.expander("Details"):
                st.text(str(e))
            st.stop()
        dur = time.time() - start

        base_name = Path(uploaded.name).stem
        st.success("Transcription complete.")
        with st.expander("Run info"):
            st.write(
                {
                    "backend": chosen,
                    "model": model_name,
                    "language": meta.get("language", ""),
                    "audio_duration_sec": meta.get("duration_sec", ""),
                    "elapsed_sec": f"{dur:.2f}",
                }
            )

        if want_lrc and lrc_text:
            if title_tag or artist_tag:
                header = []
                if artist_tag:
                    header.append(f"[ar:{artist_tag}]")
                if title_tag:
                    header.append(f"[ti:{title_tag}]")
                lrc_text = "\n".join(header + [lrc_text])
            st.code(lrc_text, language="text")
            st.download_button("Download .lrc", data=lrc_text, file_name=f"{base_name}.lrc", mime="text/plain")

        st.code(txt_text, language="text")
        st.download_button("Download .txt", data=txt_text, file_name=f"{base_name}.txt", mime="text/plain")

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
