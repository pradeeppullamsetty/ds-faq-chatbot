"""
Free voice synthesis for the Streamlit chatbot using edge-tts.

Important:
- No API keys required.
- We generate an MP3 to a temporary file, then return bytes to Streamlit (so Cloud can play it).
"""

from __future__ import annotations

import tempfile
import os
from typing import Optional

try:
    import edge_tts  # type: ignore

    EDGE_AVAILABLE = True
except Exception:
    EDGE_AVAILABLE = False


DEFAULT_VOICE_LOCALE = "en-IN"
DEFAULT_RATE = "+20%"
DEFAULT_VOLUME = "+0%"  # keep non-zero when speaking, app can mute via empty text/volume strategy
DEFAULT_PITCH = "+10Hz"


def get_audio_bytes(
    text: str,
    *,
    locale: str = DEFAULT_VOICE_LOCALE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
    pitch: str = DEFAULT_PITCH,
) -> Optional[bytes]:
    """Synthesize `text` to MP3 bytes. Returns None on failure."""
    if not text or not EDGE_AVAILABLE:
        return None

    # edge-tts API is async internally but can run synchronously via communicate.save.
    try:
        # Temporary file is used only for the duration of the request.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        communicate = edge_tts.Communicate(text, locale, rate=rate, volume=volume)
        # edge-tts doesn't accept pitch directly in some versions; keep pitch as a best-effort workaround
        # by including it in the SSML if supported. If not supported, generation still works.
        try:
            # Try SSML mode with pitch only if edge-tts accepts it.
            ssml = f"<speak><prosody pitch='{pitch}'>{text}</prosody></speak>"
            communicate = edge_tts.Communicate(ssml, locale, rate=rate, volume=volume)
        except Exception:
            pass

        communicate.save(tmp_path)

        with open(tmp_path, "rb") as af:
            data = af.read()

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return data
    except Exception:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def is_voice_available() -> bool:
    return EDGE_AVAILABLE


# Backwards-compatible alias for any older wiring
def speak(text: str) -> None:
    """Generate audio for side effects (best-effort). Kept for compatibility."""
    _ = get_audio_bytes(text)
