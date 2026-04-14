import os
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "mobilenetv3_best.pth")
REALESRGAN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "realesr-general-x4v3.pth",
)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SESSION_STORE_DIR = os.path.join(BASE_DIR, "backend", "session_store")
DEFAULT_FIREBASE_ADMIN_CREDENTIALS_PATH = os.path.join(
    BASE_DIR,
    "secrets",
    "cropguardai-71d28-firebase-adminsdk-fbsvc-8afa509ee7.json",
)
FIREBASE_ADMIN_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_ADMIN_CREDENTIALS_PATH",
    DEFAULT_FIREBASE_ADMIN_CREDENTIALS_PATH,
)

FIREBASE_WEB_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyCzDJTfg61kxfaW95XzHEPXU77pZ4rq9XA"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "cropguardai-71d28.firebaseapp.com"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "cropguardai-71d28"),
    "storageBucket": os.getenv(
        "FIREBASE_STORAGE_BUCKET",
        "cropguardai-71d28.firebasestorage.app",
    ),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "688737400665"),
    "appId": os.getenv(
        "FIREBASE_APP_ID",
        "1:688737400665:web:494dfa0e695d01e153be48",
    ),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "G-XJ5ZHS52PC"),
}
FIREBASE_ENABLED_PROVIDERS = ["google", "password"]

IMG_SIZE = 224
CONF_HIGH = 0.85
CONF_LOW = 0.65
ENHANCEMENT_TRIGGER_CONFIDENCE = 0.75
ENHANCEMENT_TRIGGER_MARGIN = 0.15
ENHANCEMENT_MIN_GAIN = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
