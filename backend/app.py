import os
import shutil
import uuid
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import FIREBASE_ENABLED_PROVIDERS, FIREBASE_WEB_CONFIG
from backend.firebase_auth import verify_id_token
from backend.inference import predict
from backend.llm import SUPPORTED_LANGUAGES, generate_response
from backend.session_store import (
    clear_user_session,
    load_user_session,
    save_user_session,
)
from backend.stt import speech_to_text
from backend.translator import get_page_translations, translate_ui_text

app = FastAPI(title="CropGuard AI")
limiter = Limiter(key_func=lambda request: get_rate_limit_key(request))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MIN_VALID_CONFIDENCE = 0.60
MIN_VALID_MARGIN = 0.10
MIN_GREEN_RATIO = 0.20
MIN_MISMATCH_CONFIDENCE = 0.80
MIN_MISMATCH_MARGIN = 0.20
MIN_LARGEST_GREEN_COMPONENT_RATIO = 0.08


def build_languages_list():
    return [
        {"code": code, "name": info["name"]}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]


def detect_source_page(request: Request) -> str:
    referer = request.headers.get("referer", "")
    if not referer:
        return "home"

    try:
        path = urlparse(referer).path or "/"
    except Exception:
        return "home"

    if path.startswith("/guide"):
        return "guide"
    if path.startswith("/result"):
        return "result"
    return "home"


def get_auth_token_from_request(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


def get_authenticated_user(request: Request, token_override: str | None = None):
    token = token_override or get_auth_token_from_request(request)
    return verify_id_token(token)


def get_rate_limit_key(request: Request) -> str:
    user_uid = get_user_uid_or_none(request)
    if user_uid:
        return f"user:{user_uid}"
    return f"ip:{get_remote_address(request)}"


def json_auth_error(message: str = "Authentication required"):
    return JSONResponse({"error": message}, status_code=401)


def get_user_uid_or_none(request: Request, token_override: str | None = None) -> str | None:
    user = get_authenticated_user(request, token_override)
    if user is None:
        return None
    return user.get("uid")


def build_page_context(
    request: Request,
    *,
    active_page: str,
    selected_language: str = "en",
    upload_error: str | None = None,
    extra: dict | None = None,
):
    context = {
        "request": request,
        "languages": build_languages_list(),
        "selected_language": selected_language,
        "active_page": active_page,
        "upload_error": upload_error,
        "firebase_config": FIREBASE_WEB_CONFIG,
        "firebase_providers": FIREBASE_ENABLED_PROVIDERS,
    }
    if extra:
        context.update(extra)
    return context


def validate_prediction(prediction: dict) -> str | None:
    top_k = prediction.get("top_k", [])
    top1 = (
        float(top_k[0]["prob"])
        if len(top_k) > 0
        else float(prediction.get("confidence", 0))
    )
    top2 = float(top_k[1]["prob"]) if len(top_k) > 1 else 0.0
    margin = top1 - top2

    if prediction.get("crop_mismatch"):
        # Only show "wrong crop selected" when the model is very sure.
        # Ambiguous cases are treated as invalid/non-crop images.
        if top1 >= MIN_MISMATCH_CONFIDENCE and margin >= MIN_MISMATCH_MARGIN:
            actual_crop = prediction.get("predicted_crop", "another crop")
            return (
                f"Wrong crop selected. This image looks like {actual_crop}. "
                "Please choose the correct crop category."
            )
        return "Wrong type of picture. Please upload a clear crop leaf image."

    if top1 < MIN_VALID_CONFIDENCE or margin < MIN_VALID_MARGIN:
        return "Wrong type of picture. Please upload a clear crop leaf image."
    return None


def is_likely_leaf_image(image_path: str) -> bool:
    """
    Quick pre-check for obvious non-leaf photos before model inference.
    """
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.uint8)
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)

        green_mask = (g > 60) & (g > r * 1.10) & (g > b * 1.10)
        green_ratio = float(np.mean(green_mask))
        if green_ratio < MIN_GREEN_RATIO:
            return False

        # Require one large connected green region, not random green noise.
        largest_ratio = largest_connected_component_ratio(green_mask)
        return largest_ratio >= MIN_LARGEST_GREEN_COMPONENT_RATIO
    except Exception:
        return False


def largest_connected_component_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    largest = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            if size > largest:
                largest = size

    return float(largest) / float(h * w)


def render_upload_error(
    request: Request,
    source_page: str,
    message: str,
    language: str,
) -> HTMLResponse:
    base_context = build_page_context(
        request,
        active_page=source_page if source_page in {"guide", "result"} else "home",
        selected_language=language,
        upload_error=message,
    )

    if source_page == "guide":
        return templates.TemplateResponse(
            "guide.html",
            base_context,
        )

    if source_page == "result":
        user_uid = get_user_uid_or_none(request)
        if user_uid:
            session_data = load_user_session(user_uid)
            saved_result_context = dict(session_data.get("result_context", {}))
            if saved_result_context:
                saved_result_context.update(base_context)
                saved_result_context["active_page"] = "result"
                return templates.TemplateResponse("result.html", saved_result_context)

    return templates.TemplateResponse(
        "index.html",
        {**base_context, "active_page": "home"},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        build_page_context(request, active_page="home"),
    )


@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse("sw.js", media_type="application/javascript")


@app.get("/offline.html")
def offline():
    return FileResponse("offline.html", media_type="text/html")


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse(
        "about.html",
        build_page_context(request, active_page="about"),
    )


@app.get("/guide", response_class=HTMLResponse)
def guide(request: Request):
    return templates.TemplateResponse(
        "guide.html",
        build_page_context(request, active_page="guide"),
    )


@app.post("/upload", response_class=HTMLResponse)
@limiter.limit("10/10minutes")
async def upload_image(
    request: Request,
    crop: str = Form(...),
    image: UploadFile = File(...),
    language: str = Form(default="en"),
    auth_token: str = Form(default=""),
):
    source_page = detect_source_page(request)
    user_uid = get_user_uid_or_none(request, auth_token)

    if user_uid is None:
        return render_upload_error(
            request,
            source_page,
            "Please sign in to upload and analyze crop images.",
            language,
        )

    ext = os.path.splitext(image.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Reject obvious non-crop images before spending inference time.
    if not is_likely_leaf_image(image_path):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
        return render_upload_error(
            request,
            source_page,
            "Wrong type of picture. Please upload a clear crop leaf image.",
            language,
        )

    prediction = predict(image_path, selected_crop=crop)

    print("\nDEBUG - Upload Image:")
    print(f"  Selected crop: {crop}")
    print(f"  Selected language: {language}")
    print(f"  Predicted crop: {prediction.get('predicted_crop')}")
    print(f"  Predicted disease: {prediction.get('predicted_disease')}")
    print(f"  Crop mismatch flag: {prediction.get('crop_mismatch')}")
    print(f"  Confidence: {prediction.get('confidence')}")

    validation_error = validate_prediction(prediction)
    if validation_error:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
        return render_upload_error(request, source_page, validation_error, language)

    final_crop_mismatch = prediction.get("crop_mismatch", False)

    confidence_pct = int(prediction["confidence"] * 100)

    if prediction.get("low_confidence"):
        llm_response = (
            "<strong>Low Confidence Prediction:</strong><br>"
            "The AI is not fully confident about this diagnosis.<br><br>"
            "<strong>Suggestion:</strong><br>"
            "Upload a clearer image with a plain background and full leaf visible."
        )
    else:
        llm_response = generate_response(prediction, language=language)

    result_context = build_page_context(
        request,
        active_page="result",
        selected_language=language,
        extra={
            "crop": prediction["predicted_crop"],
            "disease": prediction["predicted_disease"],
            "confidence": confidence_pct,
            "image_url": f"/uploads/{filename}",
            "llm_response": llm_response,
            "top_k": prediction.get("top_k", [])[:3],
            "crop_mismatch": final_crop_mismatch,
            "selected_crop": crop,
            "user_uid": user_uid,
        },
    )
    save_user_session(
        user_uid,
        {
            "prediction": prediction,
            "selected_crop": crop,
            "selected_language": language,
            "crop_mismatch": final_crop_mismatch,
            "result_context": dict(result_context),
            "chat_messages": [],
        },
    )

    return templates.TemplateResponse("result.html", result_context)


@app.post("/regenerate")
@limiter.limit("10/10minutes")
async def regenerate(request: Request, payload: dict):
    user_uid = get_user_uid_or_none(request)
    if user_uid is None:
        return json_auth_error("Please sign in to continue.")

    session_data = load_user_session(user_uid)
    prediction = session_data.get("prediction")

    if prediction is None:
        return JSONResponse({"error": "No prediction context found"}, status_code=400)

    language = payload.get("language", "en")

    if session_data.get("crop_mismatch"):
        return {
            "response": (
                "<strong>Crop Mismatch Detected:</strong><br>"
                "Please upload an image that matches the selected crop."
            )
        }

    llm_response = generate_response(prediction=prediction, language=language)

    return {"response": llm_response}


@app.post("/chat")
@limiter.limit("20/10minutes")
async def chat(request: Request, payload: dict):
    user_uid = get_user_uid_or_none(request)
    if user_uid is None:
        return json_auth_error("Please sign in to use the assistant.")

    session_data = load_user_session(user_uid)
    prediction = session_data.get("prediction")

    if prediction is None:
        return JSONResponse(
            {"reply": "No prediction context available"}, status_code=400
        )

    if session_data.get("crop_mismatch"):
        return {
            "reply": "The image does not match the selected crop. Please upload a correct image."
        }

    message = payload.get("message", "")
    language = payload.get("language", "en")
    chat_messages = list(session_data.get("chat_messages", []))
    chat_messages.append({"role": "user", "content": message})

    llm_response = generate_response(
        prediction=prediction, chat_history=chat_messages, language=language
    )
    chat_messages.append({"role": "assistant", "content": llm_response})
    session_data["chat_messages"] = chat_messages
    save_user_session(user_uid, session_data)

    return {"reply": llm_response}


@app.post("/stt")
@limiter.limit("10/10minutes")
async def speech_input(
    request: Request,
    audio: UploadFile = File(...),
):
    user = get_authenticated_user(request)
    if user is None:
        return json_auth_error("Please sign in to use voice input.")

    audio_bytes = await audio.read()
    return speech_to_text(audio_bytes)


@app.post("/translate")
async def translate_page(payload: dict):
    """
    Translate UI elements for a specific page.
    Endpoint for getting LLM-powered UI translations with caching.
    """
    page = payload.get("page", "common")
    language = payload.get("language", "en")

    try:
        translations = get_page_translations(page, language)
        return {"translations": translations, "language": language, "page": page}
    except Exception as e:
        print(f"Translation endpoint error: {e}")
        return JSONResponse(
            {"error": "Translation failed", "message": str(e)}, status_code=500
        )


@app.post("/translate-ui")
@limiter.limit("30/10minutes")
async def translate_ui(request: Request, payload: dict):
    """
    Translate a list of English UI strings to the selected language using LLM.
    """
    language = payload.get("language", "en")
    texts = payload.get("texts", [])

    if not isinstance(texts, list):
        return JSONResponse({"error": "Invalid 'texts' payload"}, status_code=400)

    normalized_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not normalized_texts:
        return {"translations": {}, "language": language}

    deduped_texts = list(dict.fromkeys(normalized_texts))
    keyed_input = {f"t{idx}": text for idx, text in enumerate(deduped_texts)}

    translated = translate_ui_text(keyed_input, language, max_length=120)
    mapping = {
        keyed_input[key]: value for key, value in translated.items() if key in keyed_input
    }

    return {"translations": mapping, "language": language}
