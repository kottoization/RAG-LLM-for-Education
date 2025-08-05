"""RAG service for embedding documents and retrieving context."""

from __future__ import annotations

from typing import Optional, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader


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

    def ingest_paths(self, paths: list[str]) -> None:
        """Embed documents from ``paths`` into the vector store and persist."""
        store = self._get_vectorstore()
        documents = []
        for path in paths:
            loader = UnstructuredFileLoader(path)
            documents.extend(loader.load())
        if documents:
            store.add_documents(documents)
            store.persist()


_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return a module-level singleton instance of :class:`RAGService`."""
    global _instance
    if _instance is None:
        _instance = RAGService()
    return _instance
