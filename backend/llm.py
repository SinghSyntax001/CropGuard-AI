import time

from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_CHAT_MODEL
from backend.logging_utils import get_logger

client = Groq(api_key=GROQ_API_KEY)
logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.85

# Language options exposed in the UI and reused across backend endpoints.
SUPPORTED_LANGUAGES = {
    "en": {"code": "en", "name": "English"},
    "hi": {"code": "hi", "name": "Hindi (हिंदी)"},
    "bn": {"code": "bn", "name": "Bengali (বাংলা)"},
    "ta": {"code": "ta", "name": "Tamil (தமிழ்)"},
    "te": {"code": "te", "name": "Telugu (తెలుగు)"},
    "mr": {"code": "mr", "name": "Marathi (मराठी)"},
    "gu": {"code": "gu", "name": "Gujarati (ગુજરાતી)"},
    "pa": {"code": "pa", "name": "Punjabi (ਪੰਜਾਬੀ)"},
    "kn": {"code": "kn", "name": "Kannada (ಕನ್ನಡ)"},
    "ml": {"code": "ml", "name": "Malayalam (മലയാളം)"},
}


def generate_response(prediction: dict, chat_history=None, language="en"):
    """
    Build farmer-friendly advice from model prediction and optional chat history.
    """
    crop = prediction.get("predicted_crop", "")
    disease = prediction.get("predicted_disease", "")
    confidence = prediction.get("confidence", 0)
    crop_mismatch = prediction.get("crop_mismatch", False)

    logger.info(
        "llm.generate_response start crop=%s disease=%s confidence=%.4f crop_mismatch=%s language=%s chat_history=%d",
        crop,
        disease,
        confidence,
        crop_mismatch,
        language,
        len(chat_history) if chat_history else 0,
    )

    language_info = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["en"])
    language_name = language_info["name"]

    # Keep output HTML simple so frontend rendering stays predictable.
    system_prompt = f"""You are an expert agricultural assistant providing guidance to farmers.
Respond ONLY in {language_name}.
Your answer must be formatted in HTML using ONLY these tags: 
<h3> for headings, <strong> for bold, <em> for emphasis, <p> for paragraphs, <br> for line breaks.
Do not use any other HTML tags, markdown, bullet points, or numbered lists.
Structure your response clearly with proper spacing between sections."""

    if crop_mismatch:
        user_prompt = f"""The uploaded leaf does not match the selected crop type. 
Please inform the user that they should select the correct crop or upload a clearer image that matches their selection.
Keep your response brief and helpful."""
    elif "healthy" in disease.lower():
        user_prompt = f"""The {crop} leaf appears to be healthy with {confidence:.2f} confidence.
Briefly reassure the user and recommend best practices for maintaining crop health.
Include practical tips that farmers can implement easily."""
    else:
        user_prompt = f"""A {crop} leaf shows symptoms of {disease} with a confidence of {confidence:.2f}.
Provide detailed information in this exact structure:
1. First, explain what {disease} is
2. Describe the visible symptoms to look for
3. Provide a step-by-step treatment plan
4. Suggest preventive measures for the future
Keep the language simple, clear and practical for farmers.
Use proper spacing between each section."""

    messages = [{"role": "system", "content": system_prompt.strip()}]

    if chat_history:
        # Keep context short to control latency and token cost.
        recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_prompt.strip()})
    else:
        messages.append({"role": "user", "content": user_prompt.strip()})

    try:
        started_at = time.perf_counter()
        logger.info(
            "llm.request model=%s max_tokens=%d messages=%d",
            GROQ_CHAT_MODEL,
            800,
            len(messages),
        )
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        result = response.choices[0].message.content.strip()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "llm.response model=%s elapsed_ms=%.1f prompt_tokens=%s completion_tokens=%s total_tokens=%s response_chars=%d",
                GROQ_CHAT_MODEL,
                elapsed_ms,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
                len(result),
            )
        else:
            logger.info(
                "llm.response model=%s elapsed_ms=%.1f response_chars=%d",
                GROQ_CHAT_MODEL,
                elapsed_ms,
                len(result),
            )
        return result
    except Exception:
        logger.exception("llm.error model=%s", GROQ_CHAT_MODEL)
        # Fallback text helps keep the app usable during API outages.
        fallback_responses = {
            "en": "<h3>Assistant Response</h3><p><strong>Note:</strong> I'm having trouble connecting to the AI assistant right now.</p><p>Based on the diagnosis, here's what you can do:</p><p>1. Isolate affected plants<br>2. Use organic fungicides<br>3. Ensure proper spacing for air circulation<br>4. Water at the base, not on leaves</p>",
            "hi": "<h3>सहायक प्रतिक्रिया</h3><p><strong>नोट:</strong> मैं अभी AI सहायक से कनेक्ट नहीं हो पा रहा हूं।</p><p>निदान के आधार पर, आप यह कर सकते हैं:</p><p>1. प्रभावित पौधों को अलग करें<br>2. जैविक कवकनाशी का प्रयोग करें<br>3. हवा के संचार के लिए उचित दूरी सुनिश्चित करें<br>4. पत्तियों पर नहीं, बल्कि जड़ों में पानी दें</p>",
            "ta": "<h3>உதவியாளர் பதில்</h3><p><strong>குறிப்பு:</strong> நான் இப்போது AI உதவியாளருடன் இணைக்க முடியவில்லை.</p><p>நோயறிதலின் அடிப்படையில், நீங்கள் இதைச் செய்யலாம்:</p><p>1. பாதிக்கப்பட்ட தாவரங்களை தனிமைப்படுத்தவும்<br>2. கரிம பூஞ்சைக்கொல்லிகளைப் பயன்படுத்தவும்<br>3. காற்று சுழற்சிக்கு சரியான இடைவெளி உறுதி செய்யவும்<br>4. இலைகளில் அல்ல, அடிவாரத்தில் தண்ணீர் ஊற்றவும்</p>",
        }
        return fallback_responses.get(language, fallback_responses["en"])
