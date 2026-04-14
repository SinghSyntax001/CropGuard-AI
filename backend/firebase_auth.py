import os
from typing import Any

from backend.config import FIREBASE_ADMIN_CREDENTIALS_PATH

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except ImportError:
    firebase_admin = None
    auth = None
    credentials = None

firebase_app = None
firebase_init_attempted = False


def get_firebase_app():
    global firebase_app, firebase_init_attempted

    if firebase_app is not None:
        return firebase_app

    if firebase_init_attempted:
        return None

    firebase_init_attempted = True

    if firebase_admin is None or auth is None or credentials is None:
        print("DEBUG - firebase-admin is not installed, auth verification disabled.")
        return None

    if not os.path.exists(FIREBASE_ADMIN_CREDENTIALS_PATH):
        print("DEBUG - Firebase Admin credentials not found, auth verification disabled.")
        return None

    try:
        credential = credentials.Certificate(FIREBASE_ADMIN_CREDENTIALS_PATH)
        firebase_app = firebase_admin.initialize_app(credential)
        print("DEBUG - Firebase Admin initialized successfully.")
    except Exception as exc:
        print(f"DEBUG - Firebase Admin initialization failed: {exc}")
        firebase_app = None

    return firebase_app


def verify_id_token(id_token: str | None) -> dict[str, Any] | None:
    if not id_token:
        return None

    app = get_firebase_app()
    if app is None:
        return None

    try:
        return auth.verify_id_token(id_token, app=app)
    except Exception as exc:
        print(f"DEBUG - Firebase token verification failed: {exc}")
        return None
