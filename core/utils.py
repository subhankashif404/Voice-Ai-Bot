from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from textblob import TextBlob
import librosa
import numpy as np

GTTS_SUPPORTED = {
    'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'ja',
    'ko', 'zh', 'zh-cn', 'zh-tw', 'ar', 'hi', 'tr', 'sv', 'da',
    'fi', 'nb', 'el', 'cs', 'sk', 'ro', 'hu', 'uk', 'id', 'ms',
    'th', 'vi', 'bn', 'ur', 'fa'
}

LANG_NAMES = {
    'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish',
    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
    'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish', 'sv': 'Swedish',
    'da': 'Danish', 'fi': 'Finnish', 'nb': 'Norwegian Bokmål',
    'el': 'Greek', 'cs': 'Czech', 'sk': 'Slovak', 'ro': 'Romanian',
    'hu': 'Hungarian', 'uk': 'Ukrainian', 'id': 'Indonesian', 'ms': 'Malay',
    'th': 'Thai', 'vi': 'Vietnamese', 'bn': 'Bengali', 'ur': 'Urdu', 'fa': 'Persian'
}

def get_language_name(code):
    return LANG_NAMES.get(code.lower(), code)

def get_language_code(name):
    for code, lang_name in LANG_NAMES.items():
        if lang_name.lower() == name.lower():
            return code
    return "en"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def safe_lang_for_gtts(lang):
    if lang in GTTS_SUPPORTED:
        return lang
    base = lang.split('-')[0]
    if base in GTTS_SUPPORTED:
        return base
    return "en"

def translate_to_english(text, lang):
    if lang == "en":
        return text
    try:
        # Using a default timeout is often better for deep-translator
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation to English error: {e}")
        return text

def translate_back(text, lang):
    if lang == "en":
        return text
    try:
        if not lang or lang == "en": return text
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except Exception as e:
        print(f"Translation back error: {e}")
        return text

def generate_ai_response(chatbot, text):
    result = chatbot(
        text, 
        max_length=75, 
        do_sample=False, 
        repetition_penalty=1.2,
        top_k=50
    )
    res = result[0]["generated_text"].strip()
    
    if not res:
        res = "I hear you! Tell me more about that."
    return res

def analyze_sentiment(text):
    blob = TextBlob(text)
    score = round(blob.sentiment.polarity, 2)
    if score >= 0.05:
        return "Positive 😊", score
    elif score <= -0.05:
        return "Negative 😞", score
    elif score == 0.0:
        return "Neutral / Calm 😌", score
    else:
        return "Neutral 😐", score

def fake_voice_detection(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=10)
        
        quiet_parts = np.abs(y[y < np.percentile(y, 10)])
        noise_floor = np.mean(quiet_parts) if len(quiet_parts) > 0 else 0
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_var = np.var(mfcc)

        if noise_floor < 1e-4 or zcr < 0.04 or mfcc_var < 100:
            return "⚠️ Possibly Synthetic / Fake Voice"
        return "✅ Likely Real Human Voice"
    except Exception:
        return "❓ Could not analyse voice"