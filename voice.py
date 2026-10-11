"""
Free voice synthesis for the Streamlit chatbot.

1st choice: edge-tts (Microsoft neural voices, no API key).
2nd choice: gTTS (Google, no API key) if edge-tts fails on the host.
Returns MP3 bytes that Streamlit can play with st.audio(...).
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
from typing import Optional

try:
    import edge_tts  # type: ignore
    EDGE_AVAILABLE = True
except Exception:
    EDGE_AVAILABLE = False

try:
    from gtts import gTTS  # type: ignore
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False


DEFAULT_VOICE_LOCALE = "en-IN"
DEFAULT_RATE = "+10%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH = "+0Hz"

# edge-tts needs a full voice name, NOT a locale like "en-IN".
VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "en-in": "en-IN-NeerjaNeural",
    "en-us": "en-US-JennyNeural",
    "ta": "ta-IN-PallaviNeural",
    "ta-in": "ta-IN-PallaviNeural",
}


def _voice_name(locale: str) -> str:
    if "Neural" in locale:  # already a full voice name
        return locale
    return VOICE_MAP.get(locale.lower(), "en-IN-NeerjaNeural")


def _run_async(coro) -> None:
    """Run a coroutine safely even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    err = []

    def _worker():
        try:
            asyncio.run(coro)
        except Exception as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if err:
        raise err[0]


def _edge_tts_bytes(text: str, locale: str, rate: str, volume: str, pitch: str) -> Optional[bytes]:
    if not EDGE_AVAILABLE:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        communicate = edge_tts.Communicate(
            text, _voice_name(locale), rate=rate, volume=volume, pitch=pitch
        )
        _run_async(communicate.save(tmp_path))  # save() is async -> must be awaited
        with open(tmp_path, "rb") as af:
            data = af.read()
        return data or None
    except Exception as e:  # noqa: BLE001
        print(f"edge-tts failed: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _gtts_bytes(text: str, locale: str) -> Optional[bytes]:
    if not GTTS_AVAILABLE:
        return None
    try:
        lang = "ta" if locale.lower().startswith("ta") else "en"
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, tld="co.in" if lang == "en" else "com").write_to_fp(buf)
        return buf.getvalue() or None
    except Exception as e:  # noqa: BLE001
        print(f"gTTS failed: {e}")
        return None


def get_audio_bytes(
    text: str,
    *,
    locale: str = DEFAULT_VOICE_LOCALE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
    pitch: str = DEFAULT_PITCH,
) -> Optional[bytes]:
    """Synthesize `text` to MP3 bytes. Returns None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    return _edge_tts_bytes(text, locale, rate, volume, pitch) or _gtts_bytes(text, locale)


def is_voice_available() -> bool:
    return EDGE_AVAILABLE or GTTS_AVAILABLE


# Backwards-compatible alias for any older wiring
def speak(text: str) -> None:
    _ = get_audio_bytes(text)
