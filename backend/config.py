import os
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _get_required_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value.strip()


MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "mobilenetv3_best.pth")
REALESRGAN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "realesr-general-x4v3.pth",
)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SESSION_STORE_DIR = os.path.join(BASE_DIR, "backend", "session_store")
FIREBASE_ADMIN_CREDENTIALS_PATH = _get_required_env("FIREBASE_ADMIN_CREDENTIALS_PATH")

FIREBASE_WEB_CONFIG = {
    "apiKey": _get_required_env("FIREBASE_API_KEY"),
    "authDomain": _get_required_env("FIREBASE_AUTH_DOMAIN"),
    "projectId": _get_required_env("FIREBASE_PROJECT_ID"),
    "storageBucket": _get_required_env("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": _get_required_env("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": _get_required_env("FIREBASE_APP_ID"),
    "measurementId": _get_required_env("FIREBASE_MEASUREMENT_ID"),
}
FIREBASE_ENABLED_PROVIDERS = [
    provider.strip()
    for provider in os.getenv("FIREBASE_ENABLED_PROVIDERS", "google,password").split(
        ","
    )
    if provider.strip()
]

GROQ_API_KEY = _get_required_env("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

IMG_SIZE = 224
CONF_HIGH = 0.85
CONF_LOW = 0.65
ENHANCEMENT_TRIGGER_CONFIDENCE = 0.75
ENHANCEMENT_TRIGGER_MARGIN = 0.15
ENHANCEMENT_MIN_GAIN = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
