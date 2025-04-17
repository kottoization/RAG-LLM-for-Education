import os
import json
from langdetect import detect

CONFIG_PATH = os.path.join("data", "user_config.json")


class LanguageHandler:
    @staticmethod
    def detect_language(text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except:
            return "en"
#TODO: check from here to the bottom
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
                return config.get("language", "en")
        return "en"

    @staticmethod
    def choose_or_detect(text: str = None) -> str:
        user_lang = LanguageHandler.get_language()
        if user_lang == "auto" and text:
            return LanguageHandler.detect_language(text)
        return user_lang
