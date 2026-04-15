import os
from typing import Any

from backend.config import FIREBASE_ADMIN_CREDENTIALS_PATH
from backend.logging_utils import get_logger

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except ImportError:
    firebase_admin = None
    auth = None
    credentials = None

firebase_app = None
firebase_init_attempted = False
logger = get_logger(__name__)


def get_firebase_app():
    global firebase_app, firebase_init_attempted

    if firebase_app is not None:
        return firebase_app

    if firebase_init_attempted:
        return None

    firebase_init_attempted = True

    if firebase_admin is None or auth is None or credentials is None:
        logger.info(
            "firebase_admin unavailable; auth verification disabled"
        )
        return None

    if not os.path.exists(FIREBASE_ADMIN_CREDENTIALS_PATH):
        logger.warning(
            "firebase_admin credentials missing path=%s",
            FIREBASE_ADMIN_CREDENTIALS_PATH,
        )
        return None

    try:
        credential = credentials.Certificate(FIREBASE_ADMIN_CREDENTIALS_PATH)
        firebase_app = firebase_admin.initialize_app(credential)
        logger.info("firebase_admin initialized successfully")
    except Exception:
        logger.exception("firebase_admin initialization failed")
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
        logger.warning("firebase token verification failed: %s", exc)
        return None
