"""Helper utilities for retrieval-augmented generation."""

from __future__ import annotations

import logging
from typing import Any


def get_context_or_empty(query: str, retriever: Any | None) -> str:
    """Return joined page contents for ``query`` using ``retriever``.

    If ``retriever`` is ``None``, retrieval fails, or no documents are found,
    an empty string is returned. This centralizes the "no documents" fallback
    behaviour so callers only append context when this function returns
    non-empty text.
    """
    if not retriever:
        return ""

    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            docs = retriever.invoke(query)
    except Exception as exc:  # pragma: no cover - retrieval errors
        logging.getLogger(__name__).warning("Retrieval failed: %s", exc)
        return ""

    if not docs:
        return ""

    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
