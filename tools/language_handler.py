import os
import json
from langdetect import detect
from deep_translator import GoogleTranslator

CONFIG_PATH = os.path.join("data", "user_config.json")

SUPPORTED_LANGUAGES = [
    "auto",
    "en",
    "pl",
    "cs",
    "sk",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "ru",
    "uk",
    "nl",
    "sv",
    "fi",
    "no",
    "da",
    "tr",
    "ja",
    "ko",
    "zh",
    "ar",
    "he",
]

LANGUAGE_LABELS = {
    "auto": "🌐 Auto-detect",
    "en": "🇺🇸 English",
    "pl": "🇵🇱 Polski",
    "cs": "🇨🇿 Čeština",
    "sk": "🇸🇰 Slovenčina",
    "de": "🇩🇪 Deutsch",
    "fr": "🇫🇷 Français",
    "es": "🇪🇸 Español",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ru": "🇷🇺 Русский",
    "uk": "🇺🇦 Українська",
    "nl": "🇳🇱 Nederlands",
    "sv": "🇸🇪 Svenska",
    "fi": "🇫🇮 Suomi",
    "no": "🇳🇴 Norsk",
    "da": "🇩🇰 Dansk",
    "tr": "🇹🇷 Türkçe",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
    "zh": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "he": "🇮🇱 עברית",
}


class LanguageHandler:
    @staticmethod
    def detect_language(text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except:
            return "en"

    @staticmethod
    def set_language(lang_code: str):
        os.makedirs("data", exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"language": lang_code}, f)

    @staticmethod
    def get_language() -> str:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                config = json.load(f)
                return config.get("language", "auto")
        return "auto"

    @staticmethod
    def choose_or_detect(text: str = None) -> str:
        user_lang = LanguageHandler.get_language()
        if user_lang == "auto" and text:
            return LanguageHandler.detect_language(text)
        return user_lang

    @staticmethod
    def translate(text: str, target: str) -> str:
        """Translate text to the target language using deep-translator."""
        # TODO: improve this feature, maybe use a different model for the whole LLM
        if not text or target == "auto":
            return text
        try:
            return GoogleTranslator(source="auto", target=target).translate(text)
        except Exception:
            return text

    @staticmethod
    def ensure_language(text: str, language: str) -> str:
        """Ensure the text is in the specified language, translating if needed."""
        if language == "auto" or not text:
            return text
        detected = LanguageHandler.detect_language(text)
        if detected != language:
            return LanguageHandler.translate(text, language)
        return text

    @staticmethod
    def supported_languages() -> list[str]:
        return SUPPORTED_LANGUAGES

    @staticmethod
    def dropdown_choices() -> list[str]:
        """Return display strings for the language dropdown."""
        return [LANGUAGE_LABELS[code] for code in SUPPORTED_LANGUAGES]

    @staticmethod
    def code_from_display(display: str) -> str:
        for code, label in LANGUAGE_LABELS.items():
            if label == display:
                return code
        return "auto"
