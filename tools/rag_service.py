"""RAG service for embedding documents and retrieving context."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from hashlib import sha256


class RAGService:
    """Lazy wrapper around a Chroma vector store."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings()
        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        self._retriever_params: Optional[Tuple[int, bool]] = None

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory="data/chroma_db",
            )
        return self._vectorstore

    def get_retriever(self, k: int = 4, mmr: bool = True):
        """Return a cached retriever from the vector store."""
        params = (k, mmr)
        if self._retriever is None or self._retriever_params != params:
            search_type = "mmr" if mmr else "similarity"
            self._retriever = self._get_vectorstore().as_retriever(
                search_type=search_type, search_kwargs={"k": k}
            )
            self._retriever_params = params
        return self._retriever

    def ingest_paths(self, items: Iterable[Union[str, Document]]) -> None:
        """Embed documents from ``items`` into the vector store and persist.

        ``items`` may be file paths or :class:`~langchain_core.documents.Document`
        instances. Chunks are deduplicated using a ``doc_hash`` metadata field to
        avoid embedding the same content multiple times.
        """
        store = self._get_vectorstore()
        documents: list[Document] = []
        for item in items:
            if isinstance(item, str):
                loader = UnstructuredFileLoader(item)
                documents.extend(loader.load())
            else:
                documents.append(item)

        to_add: list[Document] = []
        seen_hashes: set[str] = set()
        for doc in documents:
            doc_hash = doc.metadata.get("doc_hash") or sha256(
                doc.page_content.encode("utf-8")
            ).hexdigest()
            doc.metadata["doc_hash"] = doc_hash
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            if not store.get(where={"doc_hash": doc_hash}, limit=1)["ids"]:
                to_add.append(doc)

        if to_add:
            store.add_documents(to_add)
            store.persist()


_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return a module-level singleton instance of :class:`RAGService`."""
    global _instance
    if _instance is None:
        _instance = RAGService()
    return _instance
