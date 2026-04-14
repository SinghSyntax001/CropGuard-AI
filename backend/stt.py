from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_STT_MODEL

client = Groq(api_key=GROQ_API_KEY)


def speech_to_text(audio_bytes: bytes):
    # Convert uploaded speech to text with language metadata.
    response = client.audio.transcriptions.create(
        file=("speech.wav", audio_bytes),
        model=GROQ_STT_MODEL,
        response_format="verbose_json",
    )

    return {"text": response.text.strip(), "language": response.language}
