import os
import json
from langdetect import detect
import langid
from deep_translator import GoogleTranslator

CONFIG_PATH = os.path.join("data", "user_config.json")

SUPPORTED_LANGUAGES = [
    "auto", "en", "pl", "cs", "sk", "de", "fr", "es", "it", "pt",
    "ru", "uk", "nl", "sv", "fi", "no", "da", "tr", "ja", "ko", "zh", "ar", "he"
]

LANGUAGE_LABELS = {
    "auto": "\U0001F310 Auto-detect",
    "en": "\U0001F1FA\U0001F1F8 English",
    "pl": "\U0001F1F5\U0001F1F1 Polski",
    "cs": "\U0001F1E8\U0001F1FF \u010Ce\u0161tina",
    "sk": "\U0001F1F8\U0001F1F0 Sloven\u010Dina",
    "de": "\U0001F1E9\U0001F1EA Deutsch",
    "fr": "\U0001F1EB\U0001F1F7 Fran\u00E7ais",
    "es": "\U0001F1EA\U0001F1F8 Espa\u00F1ol",
    "it": "\U0001F1EE\U0001F1F9 Italiano",
    "pt": "\U0001F1F5\U0001F1F9 Portugu\u00EAs",
    "ru": "\U0001F1F7\U0001F1FA \u0420\u0443\u0441\u0441\u043A\u0438\u0439",
    "uk": "\U0001F1FA\U0001F1E6 \u0423\u043A\u0440\u0430\u0457\u043D\u0441\u044C\u043A\u0430",
    "nl": "\U0001F1F3\U0001F1F1 Nederlands",
    "sv": "\U0001F1F8\U0001F1EA Svenska",
    "fi": "\U0001F1EB\U0001F1EE Suomi",
    "no": "\U0001F1F3\U0001F1F4 Norsk",
    "da": "\U0001F1E9\U0001F1F0 Dansk",
    "tr": "\U0001F1F9\U0001F1F7 T\u00FCrk\u00E7e",
    "ja": "\U0001F1EF\U0001F1F5 \u65E5\u672C\u8A9E",
    "ko": "\U0001F1F0\U0001F1F7 \uD55C\uAD6D\uC5B4",
    "zh": "\U0001F1E8\U0001F1F3 \u4E2D\u6587",
    "ar": "\U0001F1F8\U0001F1E6 \u0627\u0644\u0639\u0631\u0628\u064A\u0629",
    "he": "\U0001F1EE\U0001F1F1 \u05E2\u05D1\u05E8\u05D9\u05EA",
}


class LanguageHandler:
    @staticmethod
    def detect_language(text: str) -> str:
        try:
            langid.set_languages([l for l in SUPPORTED_LANGUAGES if l != "auto"])
            lang, _ = langid.classify(text)
        except Exception:
            try:
                lang = detect(text)
            except Exception:
                lang = "en"
        return lang

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
