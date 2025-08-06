import logging

from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader

from tools.rag_service import RAGService


def test_ingest_and_retrieve(tmp_path):
    service = RAGService(embeddings=FakeEmbeddings(size=32), persist_directory=str(tmp_path))
    doc = Document(page_content="Cats are great pets")
    service.ingest_paths([doc])
    retriever = service.get_retriever()
    docs = retriever.invoke("cats")
    assert any("Cats are great pets" in d.page_content for d in docs)


def test_ingest_paths_lookup_error(monkeypatch, tmp_path, caplog):
    def bad_load(self):  # pragma: no cover - test helper
        raise LookupError("punkt not found")

    monkeypatch.setattr(UnstructuredFileLoader, "load", bad_load)
    file_path = tmp_path / "f.txt"
    file_path.write_text("test")
    service = RAGService(embeddings=FakeEmbeddings(size=32), persist_directory=str(tmp_path / "db"))

    caplog.set_level(logging.ERROR)
    msg = service.ingest_paths([str(file_path)])
    assert "nltk.download('punkt')" in msg
    assert "Failed to load" in caplog.text
