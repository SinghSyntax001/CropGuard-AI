import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from backend.config import (
    DEVICE,
    ENHANCEMENT_MIN_GAIN,
    ENHANCEMENT_TRIGGER_CONFIDENCE,
    ENHANCEMENT_TRIGGER_MARGIN,
    IMG_SIZE,
    MODEL_PATH,
    REALESRGAN_MODEL_PATH,
)
from backend.logging_utils import get_logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = DEVICE
enhancer = None
enhancer_load_attempted = False
logger = get_logger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError:
    RRDBNet = None
    RealESRGANer = None

# Load the classifier once at startup.
checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

logger.info("inference model loaded classes=%d", NUM_CLASSES)
logger.debug("inference class list=%s", CLASS_NAMES)

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES,
)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# Match validation-time preprocessing used during training.
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def get_enhancer():
    global enhancer, enhancer_load_attempted

    if enhancer is not None:
        return enhancer

    if enhancer_load_attempted:
        return None

    enhancer_load_attempted = True

    if cv2 is None or RRDBNet is None or RealESRGANer is None:
        logger.info("Real-ESRGAN dependencies unavailable; enhancement disabled")
        return None

    if not os.path.exists(REALESRGAN_MODEL_PATH):
        logger.warning(
            "Real-ESRGAN weights missing path=%s; enhancement disabled",
            REALESRGAN_MODEL_PATH,
        )
        return None

    try:
        rrdbnet = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        enhancer = RealESRGANer(
            scale=4,
            model_path=REALESRGAN_MODEL_PATH,
            model=rrdbnet,
            device=device,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=device == "cuda",
        )
        logger.info("Real-ESRGAN loaded successfully")
    except Exception:
        logger.exception("Failed to load Real-ESRGAN")
        enhancer = None

    return enhancer


def parse_crop_name(label: str) -> str:
    label_lower = label.lower()

    if "_" in label:
        return label.split("_")[0].lower()

    if "healthy" in label_lower:
        crop_name = label_lower.replace("healthy", "").strip().rstrip()
        return crop_name or "unknown"

    return label_lower


def split_prediction_label(predicted_class: str) -> tuple[str, str]:
    if "_" in predicted_class:
        return predicted_class.split("_", 1)

    if "healthy" in predicted_class.lower():
        parts = predicted_class.lower().split("healthy")
        if len(parts) > 1 and parts[0].strip():
            return parts[0].strip().capitalize(), "Healthy"
        return "Unknown", "Healthy"

    return predicted_class, "Unknown"


def summarize_prediction(
    probs: torch.Tensor,
    raw_class: str,
    raw_conf: float,
    selected_crop: str | None,
    top_k: int,
) -> dict:
    filtered = []
    crop_mismatch = False

    if selected_crop:
        selected_crop_lower = selected_crop.lower()
        raw_crop = parse_crop_name(raw_class)

        logger.debug(
            "summarize_prediction selected_crop=%s raw_crop=%s",
            selected_crop_lower,
            raw_crop,
        )

        if raw_crop != selected_crop_lower and raw_crop != "unknown":
            logger.warning(
                "crop mismatch candidate raw_crop=%s selected_crop=%s",
                raw_crop,
                selected_crop_lower,
            )
            crop_mismatch = True

        for prob, cls in zip(probs, CLASS_NAMES):
            crop_name = parse_crop_name(cls)
            if crop_name == selected_crop_lower:
                logger.debug(
                    "summarize_prediction match class=%s crop=%s prob=%.4f",
                    cls,
                    crop_name,
                    prob.item(),
                )
                filtered.append((cls, prob.item()))

    logger.debug("summarize_prediction filtered_count=%d", len(filtered))
    if filtered:
        logger.debug("summarize_prediction filtered_items=%s", filtered)

    low_confidence = False

    if not filtered and selected_crop:
        logger.warning(
            "No matching classes found for selected crop=%s; treating as crop mismatch",
            selected_crop,
        )
        predicted_class = raw_class
        confidence = raw_conf
        crop_mismatch = True
    elif not filtered:
        predicted_class = raw_class
        confidence = raw_conf
    else:
        healthy_candidates = [
            (cls, prob) for cls, prob in filtered if "healthy" in cls.lower()
        ]
        disease_candidates = [
            (cls, prob) for cls, prob in filtered if "healthy" not in cls.lower()
        ]

        logger.debug(
            "summarize_prediction healthy_candidates=%d disease_candidates=%d",
            len(healthy_candidates),
            len(disease_candidates),
        )

        best_healthy = (
            max(healthy_candidates, key=lambda item: item[1])
            if healthy_candidates
            else None
        )
        best_disease = (
            max(disease_candidates, key=lambda item: item[1])
            if disease_candidates
            else None
        )

        if best_healthy:
            logger.debug(
                "summarize_prediction best_healthy=%s confidence=%.4f",
                best_healthy[0],
                best_healthy[1],
            )
        if best_disease:
            logger.debug(
                "summarize_prediction best_disease=%s confidence=%.4f",
                best_disease[0],
                best_disease[1],
            )

        if best_healthy and (not best_disease or best_disease[1] < 0.80):
            predicted_class, confidence = best_healthy
            low_confidence = False
            logger.debug("summarize_prediction selected=healthy_override")
        else:
            predicted_class, confidence = best_disease
            low_confidence = confidence < 0.50
            logger.debug(
                "summarize_prediction selected=disease confidence=%.4f low_confidence=%s",
                confidence,
                low_confidence,
            )

    predicted_crop, predicted_disease = split_prediction_label(predicted_class)

    if crop_mismatch:
        actual_crop = raw_class.split("_")[0] if "_" in raw_class else raw_class
        if "healthy" in raw_class.lower() and "_" not in raw_class:
            parts = raw_class.lower().split("healthy")
            actual_crop = parts[0].strip().capitalize() if parts[0].strip() else "Unknown"
        logger.warning(
            "crop mismatch detected selected_crop=%s actual_crop=%s",
            selected_crop,
            actual_crop,
        )
        predicted_crop = actual_crop

    top_probs, top_idxs = torch.topk(probs, top_k)
    top_k_results = [
        {"class": CLASS_NAMES[idx.item()], "prob": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_idxs)
    ]

    top1_prob = float(top_probs[0].item()) if len(top_probs) > 0 else float(raw_conf)
    top2_prob = float(top_probs[1].item()) if len(top_probs) > 1 else 0.0
    confidence_margin = top1_prob - top2_prob

    return {
        "predicted_class": predicted_class,
        "predicted_crop": predicted_crop,
        "predicted_disease": predicted_disease,
        "confidence": round(float(confidence), 4),
        "low_confidence": low_confidence,
        "crop_mismatch": crop_mismatch,
        "top_k": top_k_results,
        "top1_prob": round(top1_prob, 4),
        "top2_prob": round(top2_prob, 4),
        "confidence_margin": round(confidence_margin, 4),
    }


def run_model(image: Image.Image, selected_crop: str | None, top_k: int) -> dict:
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0]

    raw_conf, raw_idx = torch.max(probs, dim=0)
    raw_class = CLASS_NAMES[raw_idx.item()]

    logger.debug(
        "run_model raw_prediction class=%s confidence=%.4f",
        raw_class,
        raw_conf.item(),
    )

    summary = summarize_prediction(
        probs=probs,
        raw_class=raw_class,
        raw_conf=float(raw_conf.item()),
        selected_crop=selected_crop,
        top_k=top_k,
    )
    summary["raw_class"] = raw_class
    summary["raw_confidence"] = round(float(raw_conf.item()), 4)
    return summary


def should_try_enhancement(result: dict) -> tuple[bool, str]:
    if result.get("crop_mismatch"):
        return False, ""

    confidence = float(result.get("top1_prob", result.get("confidence", 0.0)))
    margin = float(result.get("confidence_margin", 0.0))

    if confidence < ENHANCEMENT_TRIGGER_CONFIDENCE:
        return True, "low_confidence"

    if margin < ENHANCEMENT_TRIGGER_MARGIN:
        return True, "small_margin"

    return False, ""


def enhance_image(image: Image.Image) -> Image.Image | None:
    upsampler = get_enhancer()
    if upsampler is None or cv2 is None:
        return None

    try:
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        enhanced_bgr, _ = upsampler.enhance(image_bgr, outscale=2)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)
    except Exception:
        logger.exception("Real-ESRGAN enhancement failed")
        return None


def should_keep_enhanced_result(original: dict, enhanced: dict) -> bool:
    original_conf = float(original.get("top1_prob", original.get("confidence", 0.0)))
    enhanced_conf = float(enhanced.get("top1_prob", enhanced.get("confidence", 0.0)))
    original_margin = float(original.get("confidence_margin", 0.0))
    enhanced_margin = float(enhanced.get("confidence_margin", 0.0))

    if enhanced.get("crop_mismatch") and not original.get("crop_mismatch"):
        return False

    if enhanced_conf >= original_conf + ENHANCEMENT_MIN_GAIN:
        return True

    if enhanced_conf > original_conf and enhanced_margin > original_margin:
        return True

    return False


def predict(image_path: str, selected_crop: str = None, top_k: int = 5) -> dict:
    image = Image.open(image_path).convert("RGB")

    started_at = time.perf_counter()
    logger.info("prediction start image=%s selected_crop=%s", image_path, selected_crop)

    original_result = run_model(image=image, selected_crop=selected_crop, top_k=top_k)
    original_result["enhancement_applied"] = False
    original_result["enhancement_reason"] = ""

    use_enhancement, reason = should_try_enhancement(original_result)
    if not use_enhancement:
        logger.info("prediction enhancement_skipped reason=confident_enough")
        logger.info(
            "prediction final crop=%s disease=%s confidence=%.4f crop_mismatch=%s low_confidence=%s top3=%s elapsed_ms=%.1f",
            original_result["predicted_crop"],
            original_result["predicted_disease"],
            original_result["confidence"],
            original_result["crop_mismatch"],
            original_result["low_confidence"],
            original_result["top_k"][:3],
            (time.perf_counter() - started_at) * 1000,
        )
        return original_result

    logger.info("prediction enhancement_triggered reason=%s", reason)
    enhanced_image = enhance_image(image)
    if enhanced_image is None:
        logger.info("prediction enhancement_unavailable using_original=true")
        logger.info(
            "prediction final crop=%s disease=%s confidence=%.4f crop_mismatch=%s low_confidence=%s top3=%s elapsed_ms=%.1f",
            original_result["predicted_crop"],
            original_result["predicted_disease"],
            original_result["confidence"],
            original_result["crop_mismatch"],
            original_result["low_confidence"],
            original_result["top_k"][:3],
            (time.perf_counter() - started_at) * 1000,
        )
        return original_result

    enhanced_result = run_model(
        image=enhanced_image,
        selected_crop=selected_crop,
        top_k=top_k,
    )
    enhanced_result["enhancement_applied"] = True
    enhanced_result["enhancement_reason"] = reason

    if should_keep_enhanced_result(original_result, enhanced_result):
        final_result = enhanced_result
        logger.info("prediction enhancement_improved using_enhanced_result=true")
    else:
        final_result = original_result
        logger.info("prediction enhancement_not_better keeping_original=true")

    logger.info(
        "prediction final crop=%s disease=%s confidence=%.4f crop_mismatch=%s low_confidence=%s top3=%s elapsed_ms=%.1f",
        final_result["predicted_crop"],
        final_result["predicted_disease"],
        final_result["confidence"],
        final_result["crop_mismatch"],
        final_result["low_confidence"],
        final_result["top_k"][:3],
        (time.perf_counter() - started_at) * 1000,
    )

    return final_result
