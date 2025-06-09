"""
RAGModule
---------
This module provides tools for:
- ingesting local documents (TXT/PDF) into chunks
- building and persisting a Chroma vectorstore
- performing semantic search and QA over ingested files
- easy extension for educational pipelines (e.g. retrieval + summarization)
"""

from .rag import RAGHandler
