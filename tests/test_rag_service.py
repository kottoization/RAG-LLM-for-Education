from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from tools.rag_service import RAGService


def test_ingest_and_retrieve(tmp_path):
    service = RAGService(embeddings=FakeEmbeddings(size=32), persist_directory=str(tmp_path))
    doc = Document(page_content="Cats are great pets")
    service.ingest_paths([doc])
    retriever = service.get_retriever()
    docs = retriever.get_relevant_documents("cats")
    assert any("Cats are great pets" in d.page_content for d in docs)
