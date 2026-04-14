import os

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = DEVICE
enhancer = None
enhancer_load_attempted = False

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

print(f"DEBUG: Loaded {NUM_CLASSES} classes")
print(f"DEBUG: Full class list: {CLASS_NAMES}")

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
        print("DEBUG - Real-ESRGAN dependencies are not installed, enhancement disabled.")
        return None

    if not os.path.exists(REALESRGAN_MODEL_PATH):
        print("DEBUG - Real-ESRGAN weights not found, enhancement disabled.")
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
        print("DEBUG - Real-ESRGAN loaded successfully.")
    except Exception as exc:
        print(f"DEBUG - Failed to load Real-ESRGAN: {exc}")
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

        print(f"  Looking for crop: '{selected_crop_lower}' in classes...")
        print(f"  Raw crop extracted: '{raw_crop}'")

        if raw_crop != selected_crop_lower and raw_crop != "unknown":
            print(
                f"  WARNING: Raw prediction crop '{raw_crop}' doesn't match selected '{selected_crop_lower}'"
            )
            crop_mismatch = True

        for prob, cls in zip(probs, CLASS_NAMES):
            crop_name = parse_crop_name(cls)
            if crop_name == selected_crop_lower:
                print(f"    MATCH: {cls} -> {crop_name} (prob: {prob.item():.4f})")
                filtered.append((cls, prob.item()))

    print(f"  Filtered list length: {len(filtered)}")
    if filtered:
        print(f"  Filtered items: {filtered}")

    low_confidence = False

    if not filtered and selected_crop:
        print(f"  ERROR: No matching classes found for selected crop '{selected_crop}'")
        print("  This confirms crop mismatch")
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

        print(f"  Healthy candidates: {len(healthy_candidates)}")
        print(f"  Disease candidates: {len(disease_candidates)}")

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
            print(f"  Best healthy: {best_healthy[0]} ({best_healthy[1]:.4f})")
        if best_disease:
            print(f"  Best disease: {best_disease[0]} ({best_disease[1]:.4f})")

        if best_healthy and (not best_disease or best_disease[1] < 0.80):
            predicted_class, confidence = best_healthy
            low_confidence = False
            print("  Selected: Healthy override")
        else:
            predicted_class, confidence = best_disease
            low_confidence = confidence < 0.50
            print(
                f"  Selected: Disease (confidence: {confidence:.4f}, low_confidence: {low_confidence})"
            )

    predicted_crop, predicted_disease = split_prediction_label(predicted_class)

    if crop_mismatch:
        actual_crop = raw_class.split("_")[0] if "_" in raw_class else raw_class
        if "healthy" in raw_class.lower() and "_" not in raw_class:
            parts = raw_class.lower().split("healthy")
            actual_crop = parts[0].strip().capitalize() if parts[0].strip() else "Unknown"
        print(
            f"  Crop mismatch detected! User selected '{selected_crop}' but image is '{actual_crop}'"
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

    print(f"  Raw prediction: {raw_class} (confidence: {raw_conf.item():.4f})")

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
    except Exception as exc:
        print(f"DEBUG - Real-ESRGAN enhancement failed: {exc}")
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

    print("\nDEBUG - Prediction Start:")
    print(f"  Image: {image_path}")
    print(f"  Selected crop: {selected_crop}")

    original_result = run_model(image=image, selected_crop=selected_crop, top_k=top_k)
    original_result["enhancement_applied"] = False
    original_result["enhancement_reason"] = ""

    use_enhancement, reason = should_try_enhancement(original_result)
    if not use_enhancement:
        print("  Enhancement skipped: original prediction is confident enough.")
        print(
            f"  Final prediction: {original_result['predicted_crop']} - {original_result['predicted_disease']}"
        )
        print(f"  Confidence: {original_result['confidence']:.4f}")
        print(f"  Crop mismatch flag: {original_result['crop_mismatch']}")
        print(f"  Low confidence flag: {original_result['low_confidence']}")
        print(f"  Top 3 predictions: {original_result['top_k'][:3]}")
        print("DEBUG - Prediction End\n")
        return original_result

    print(f"  Enhancement triggered: {reason}")
    enhanced_image = enhance_image(image)
    if enhanced_image is None:
        print("  Enhancement unavailable or failed, using original prediction.")
        print(
            f"  Final prediction: {original_result['predicted_crop']} - {original_result['predicted_disease']}"
        )
        print(f"  Confidence: {original_result['confidence']:.4f}")
        print(f"  Crop mismatch flag: {original_result['crop_mismatch']}")
        print(f"  Low confidence flag: {original_result['low_confidence']}")
        print(f"  Top 3 predictions: {original_result['top_k'][:3]}")
        print("DEBUG - Prediction End\n")
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
        print("  Enhancement improved the prediction, using enhanced result.")
    else:
        final_result = original_result
        print("  Enhancement did not improve enough, keeping original result.")

    print(
        f"  Final prediction: {final_result['predicted_crop']} - {final_result['predicted_disease']}"
    )
    print(f"  Confidence: {final_result['confidence']:.4f}")
    print(f"  Crop mismatch flag: {final_result['crop_mismatch']}")
    print(f"  Low confidence flag: {final_result['low_confidence']}")
    print(f"  Top 3 predictions: {final_result['top_k'][:3]}")
    print("DEBUG - Prediction End\n")

    return final_result
