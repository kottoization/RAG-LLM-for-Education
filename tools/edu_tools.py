from __future__ import annotations

from langchain.tools import tool
from nltk.corpus import wordnet as wn
from langdetect import detect
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


from RAGModule.rag import RAGHandler

_rag = None

def _get_rag() -> RAGHandler:
    global _rag
    if _rag is None:
        _rag = RAGHandler()
        try:
            _rag.load_vectorstore()
        except Exception:
            pass
    return _rag


@tool
def document_search(query: str) -> str:
    """Answer a question using documents indexed by the system."""
    rag = _get_rag()
    try:
        return rag.answer(query)
    except Exception as e:  # pragma: no cover - LLM errors
        return f"Error using RAG: {e}"


@tool
def current_date(_: str = "") -> str:
    """Return today's date in ISO format."""
    return datetime.utcnow().strftime("%Y-%m-%d")


@tool
def detect_language(text: str) -> str:
    """Detect the language of a given text sample."""
    try:
        return detect(text)
    except Exception as e:  # pragma: no cover - detection errors
        return f"Error detecting language: {e}"
