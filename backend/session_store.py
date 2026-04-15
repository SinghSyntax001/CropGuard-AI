import json
import os
from typing import Any

from backend.config import SESSION_STORE_DIR
from backend.logging_utils import get_logger

os.makedirs(SESSION_STORE_DIR, exist_ok=True)
logger = get_logger(__name__)


def session_file_path(user_uid: str) -> str:
    safe_uid = "".join(ch for ch in user_uid if ch.isalnum() or ch in {"-", "_"})
    return os.path.join(SESSION_STORE_DIR, f"{safe_uid}.json")


def load_user_session(user_uid: str) -> dict[str, Any]:
    path = session_file_path(user_uid)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        logger.exception("failed to load user session path=%s", path)
        return {}


def save_user_session(user_uid: str, payload: dict[str, Any]) -> None:
    path = session_file_path(user_uid)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except Exception:
        logger.exception("failed to save user session path=%s", path)


def clear_user_session(user_uid: str) -> None:
    path = session_file_path(user_uid)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            logger.exception("failed to clear user session path=%s", path)
