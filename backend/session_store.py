import json
import os
from typing import Any

from backend.config import SESSION_STORE_DIR

os.makedirs(SESSION_STORE_DIR, exist_ok=True)


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
        return {}


def save_user_session(user_uid: str, payload: dict[str, Any]) -> None:
    path = session_file_path(user_uid)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def clear_user_session(user_uid: str) -> None:
    path = session_file_path(user_uid)
    if os.path.exists(path):
        os.remove(path)
