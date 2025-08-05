from __future__ import annotations

from langchain.tools import tool
from nltk.corpus import wordnet as wn
from datetime import datetime

# Lazy import wikipedia to avoid unnecessary dependency at runtime
try:
    import wikipedia  # type: ignore
except Exception:  # pragma: no cover - wikipedia may not be installed
    wikipedia = None


@tool
def wikipedia_search(query: str) -> str:
    """Return a short summary for a topic from Wikipedia."""
    if wikipedia is None:
        return "wikipedia library not available."
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:  # pragma: no cover - network issues
        return f"Error fetching data from Wikipedia: {e}"


@tool
def define_word(word: str) -> str:
    """Give dictionary definitions for a word using WordNet."""
    try:
        synsets = wn.synsets(word)
        if not synsets:
            return "No definition found."
        defs = {s.definition() for s in synsets}
        return "; ".join(sorted(defs))
    except Exception as e:  # pragma: no cover
        return f"Error retrieving definition: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:  # pragma: no cover - invalid expression
        return f"Error evaluating expression: {e}"


@tool
def document_search(_: str) -> str:
    """Placeholder for document search functionality."""
    return "Document search not implemented"


@tool
def current_date(_: str = "") -> str:
    """Return today's date in ISO format."""
    return datetime.utcnow().strftime("%Y-%m-%d")


@tool
def current_weekday(_: str = "") -> str:
    """Return the current day of the week."""
    return datetime.utcnow().strftime("%A")


@tool
def detect_language(text: str) -> str:
    """Detect the language of a given text sample."""
    from tools.language_handler import LanguageHandler
    try:
        return LanguageHandler.detect_language(text)
    except Exception as e:  # pragma: no cover - detection errors
        return f"Error detecting language: {e}"
