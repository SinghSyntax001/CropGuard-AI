import time

from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_STT_MODEL
from backend.logging_utils import get_logger

client = Groq(api_key=GROQ_API_KEY)
logger = get_logger(__name__)


def speech_to_text(audio_bytes: bytes):
    # Convert uploaded speech to text with language metadata.
    started_at = time.perf_counter()
    logger.info(
        "speech_to_text request model=%s audio_bytes=%d",
        GROQ_STT_MODEL,
        len(audio_bytes),
    )
    response = client.audio.transcriptions.create(
        file=("speech.wav", audio_bytes),
        model=GROQ_STT_MODEL,
        response_format="verbose_json",
    )

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "speech_to_text response model=%s elapsed_ms=%.1f language=%s text_chars=%d",
        GROQ_STT_MODEL,
        elapsed_ms,
        getattr(response, "language", None),
        len(response.text.strip()),
    )

    return {"text": response.text.strip(), "language": response.language}
