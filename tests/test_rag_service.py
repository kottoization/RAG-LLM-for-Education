import logging
import base64

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


def test_ingest_paths_import_error(monkeypatch, tmp_path, caplog):
    def bad_load(self):  # pragma: no cover - test helper
        raise ImportError("unstructured dependency missing")

    monkeypatch.setattr(UnstructuredFileLoader, "load", bad_load)
    file_path = tmp_path / "f.txt"
    file_path.write_text("test")
    service = RAGService(embeddings=FakeEmbeddings(size=32), persist_directory=str(tmp_path / "db2"))

    caplog.set_level(logging.ERROR)
    msg = service.ingest_paths([str(file_path)])
    assert "unstructured[pdf]" in msg
    assert "Failed to load" in caplog.text


def test_ingest_pdf(tmp_path):
    pdf_b64 = (
        "JVBERi0xLjUKMSAwIG9iaiA8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4gZW5k"
        "b2JqCjIgMCBvYmogPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4g"
        "ZW5kb2JqCjMgMCBvYmogPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3gg"
        "WzAgMCAyMDAgMjAwXSAvQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAv"
        "RjEgNSAwIFIgPj4gPj4gPj4gZW5kb2JqCjQgMCBvYmogPDwgL0xlbmd0aCA0NCA+PiBzdHJl"
        "YW0KQlQgL0YxIDEyIFRmIDcyIDEwMCBUZCAoQ2F0cyBhcmUgZ3JlYXQgcGV0cykgVGogRVQK"
        "ZW5kc3RyZWFtIGVuZG9iago1IDAgb2JqIDw8IC9UeXBlIC9Gb250IC9TdWJ0eXBlIC9UeXBl"
        "MSAvQmFzZUZvbnQgL0hlbHZldGljYSA+PiBlbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2"
        "NTUzNSBmIAowMDAwMDAwMDEwIDAwMDAwIG4gCjAwMDAwMDAwNTYgMDAwMDAgbiAKMDAwMDAw"
        "MDExNCAwMDAwMCBuIAowMDAwMDAwMjQ1IDAwMDAwIG4gCjAwMDAwMDAzMzQgMDAwMDAgbiAK"
        "dHJhaWxlciA8PCAvUm9vdCAxIDAgUiAvU2l6ZSA2ID4+CnN0YXJ0eHJlZgo0MTYKJSVFT0YK"
    )
    pdf_path = tmp_path / "cats.pdf"
    pdf_path.write_bytes(base64.b64decode(pdf_b64))
    service = RAGService(embeddings=FakeEmbeddings(size=32), persist_directory=str(tmp_path / "db3"))
    err = service.ingest_paths([str(pdf_path)])
    assert err is None
    docs = service.get_retriever().invoke("cats")
    assert any("Cats are great pets" in d.page_content for d in docs)
