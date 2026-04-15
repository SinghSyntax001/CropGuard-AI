import json
import time

from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_CHAT_MODEL
from backend.logging_utils import get_logger

client = Groq(api_key=GROQ_API_KEY)
logger = get_logger(__name__)

# In-memory cache to avoid repeated translation calls for same payload.
UI_TRANSLATIONS_CACHE = {}


def translate_ui_text(
    text_dict: dict, target_lang: str, max_length: int = None
) -> dict:
    """
    Translate a mapping of UI keys -> English text to the target language.
    """
    cache_key = f"{target_lang}_{hash(frozenset(text_dict.items()))}"
    if cache_key in UI_TRANSLATIONS_CACHE:
        logger.info(
            "translate_ui_text cache_hit target_lang=%s keys=%d",
            target_lang,
            len(text_dict),
        )
        return UI_TRANSLATIONS_CACHE[cache_key]

    if target_lang == "en":
        return text_dict

    language_names = {
        "hi": "Hindi",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "kn": "Kannada",
        "ml": "Malayalam",
    }

    target_language = language_names.get(target_lang, target_lang)

    length_constraint = (
        f"\nIMPORTANT: Keep each translation under {max_length} characters to fit the UI."
        if max_length
        else ""
    )

    prompt = f"""You are a professional UI translator specializing in agricultural technology.
Translate the following User Interface labels from English to {target_language}.

RULES:
1. Keep technical agricultural and plant pathology terms accurate
2. Maintain the tone appropriate for farmers
3. Return ONLY a valid JSON object mapping keys to translations
4. Do not add any explanation or extra text{length_constraint}

Input JSON:
{json.dumps(text_dict, ensure_ascii=False, indent=2)}

Output ONLY the translated JSON:"""

    try:
        started_at = time.perf_counter()
        logger.info(
            "translate_ui_text request model=%s target_lang=%s keys=%d max_tokens=%d",
            GROQ_CHAT_MODEL,
            target_lang,
            len(text_dict),
            1500,
        )
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Return ONLY valid JSON without any markdown or explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        result_text = response.choices[0].message.content.strip()

        # Some models wrap JSON in markdown fences; strip them safely.
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        result_text = result_text.strip()

        translations = json.loads(result_text)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "translate_ui_text response model=%s target_lang=%s elapsed_ms=%.1f prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                GROQ_CHAT_MODEL,
                target_lang,
                elapsed_ms,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        else:
            logger.info(
                "translate_ui_text response model=%s target_lang=%s elapsed_ms=%.1f",
                GROQ_CHAT_MODEL,
                target_lang,
                elapsed_ms,
            )

        UI_TRANSLATIONS_CACHE[cache_key] = translations

        return translations

    except Exception:
        logger.exception("translate_ui_text error target_lang=%s", target_lang)
        return text_dict


def get_page_translations(page: str, target_lang: str) -> dict:
    """
    Return translated text bundle for one page.
    """

    ui_texts = {
        "home": {
            "title": "Detect Crop Disease Instantly",
            "subtitle": "Upload a leaf image and get AI-powered diagnosis and treatment.",
            "upload_btn": "Upload Image",
            "nav_home": "HOME",
            "nav_about": "ABOUT",
            "nav_guide": "GUIDE",
            "select_crop": "Select Crop",
            "fruits": "Fruits",
            "vegetables": "Vegetables",
            "cancel": "Cancel",
            "upload": "Upload",
            "footer": "Copyright @ 2026 CropGuard AI",
            "apple": "Apple",
            "banana": "Banana",
            "mango": "Mango",
            "grapes": "Grapes",
            "tomato": "Tomato",
            "potato": "Potato",
            "cauliflower": "Cauliflower",
            "corn": "Corn",
        },
        "about": {
            "nav_home": "HOME",
            "nav_about": "ABOUT",
            "nav_guide": "GUIDE",
            "about_title": "About CropGuard AI",
            "about_subtitle": "Empowering farmers worldwide with AI-driven plant disease detection",
            "stat_1_label": "Crops and Diseases",
            "stat_2_label": "Accuracy Rate",
            "stat_3_label": "Languages",
            "stat_4_label": "AI Analysis",
            "mission_title": "Our Mission",
            "mission_text": "CropGuard AI is revolutionizing agriculture by leveraging cutting-edge artificial intelligence and computer vision to help farmers detect crop diseases at the earliest stage, enabling timely intervention and maximizing crop yield.",
            "feature_1_title": "Smart Diagnosis",
            "feature_1_text": "Upload leaf images for instant AI-powered disease analysis powered by MobileNetV3.",
            "feature_2_title": "10+ Languages",
            "feature_2_text": "Support for major Indian languages including Hindi, Tamil, Bengali and more.",
            "feature_3_title": "Expert Guidance",
            "feature_3_text": "Get detailed treatment plans and preventive measures using AI assistance.",
            "impact_title": "Impact",
            "impact_subtitle": "Building a sustainable future for farmers",
            "impact_1_title": "55K+",
            "impact_1_label": "Training Images",
            "impact_1_text": "Comprehensive dataset for accurate predictions",
            "impact_2_title": "Real-time",
            "impact_2_label": "Analysis",
            "impact_2_text": "Get results in seconds, not hours",
            "impact_3_title": "Farmer",
            "impact_3_label": "Friendly",
            "impact_3_text": "Designed specifically for agricultural users",
            "team_title": "Meet Our Team",
            "team_subtitle": "Experts behind CropGuard AI, dedicated to transforming agriculture",
            "member_1_role": "AI and ML Engineer",
            "member_1_bio": "Deep learning specialist building accurate disease detection models using state-of-the-art neural networks.",
            "member_2_role": "UI and UX Designer",
            "member_2_bio": "Creates intuitive and beautiful interfaces that make agriculture technology accessible to all farmers.",
            "member_3_role": "Backend Developer",
            "member_3_bio": "Builds robust and scalable FastAPI backend infrastructure with seamless AI integrations.",
            "about_footer": "Copyright @ 2026 CropGuard AI",
        },
        "guide": {
            "nav_home": "HOME",
            "nav_about": "ABOUT",
            "nav_guide": "GUIDE",
            "guide_hero_title": "How to Use CropGuard AI",
            "guide_hero_subtitle": "Your step-by-step guide to diagnose crop diseases efficiently",
            "guide_title": "Step-by-Step Guide",
            "guide_intro": "Follow these steps to detect crop diseases and access helpful resources.",
            "step_1_title": "Step 1: Create an Account",
            "step_1_text": "Sign up and log in to access full features.",
            "step_2_title": "Step 2: Upload Crop Image",
            "step_2_text": "Upload a clear photo of the affected leaf.",
            "step_3_title": "Step 3: AI Disease Detection",
            "step_3_text": "AI analyzes the image in seconds.",
            "step_4_title": "Step 4: Get Treatment",
            "step_4_text": "View AI-generated treatment and prevention tips.",
            "step_5_title": "Step 5: Ask Follow-up Questions",
            "step_5_text": "Use the AI chat assistant to ask questions in your preferred language.",
            "step_6_title": "Step 6: Continuous Learning",
            "step_6_text": "Learn preventive measures and sustainable farming practices.",
            "cta_title": "Protect Your Crops Today",
            "cta_text": "Do not wait for diseases to spread. Upload your crop image now and get instant AI-powered insights.",
            "cta_upload": "Upload Image Now",
            "guide_footer": "Copyright @ 2026 CropGuard AI",
            "select_crop": "Select Crop",
            "fruits": "Fruits",
            "vegetables": "Vegetables",
            "cancel": "Cancel",
            "upload": "Upload",
            "apple": "Apple",
            "banana": "Banana",
            "mango": "Mango",
            "grapes": "Grapes",
            "tomato": "Tomato",
            "potato": "Potato",
            "cauliflower": "Cauliflower",
            "corn": "Corn",
        },
        "result": {
            "diagnosis_result": "Diagnosis Result",
            "crop_label": "Crop:",
            "disease_label": "Disease:",
            "confidence_level": "Confidence Level",
            "high_confidence": "High Confidence",
            "moderate_confidence": "Moderate Confidence",
            "low_confidence": "Low Confidence",
            "other_diagnoses": "Other Possible Diagnoses",
            "ai_assistant": "AI Assistant",
            "chat_placeholder": "Ask a follow-up question in your language...",
            "loading_response": "Loading response in selected language...",
            "loading_failed": "Unable to load response. Please try again.",
            "typing": "AI is typing...",
            "chat_error": "Sorry, I am having trouble responding. Please try again.",
            "processing_voice": "Processing voice input...",
            "recording": "Recording... Speak now",
            "processing": "Processing...",
            "back_home": "Back to Home",
        },
        "common": {
            "loading": "Loading...",
            "error": "Error occurred",
            "try_again": "Please try again",
            "language": "Language",
        },
    }

    page_texts = dict(ui_texts.get(page, {}))

    page_texts.update(ui_texts.get("common", {}))

    return translate_ui_text(page_texts, target_lang, max_length=70)
