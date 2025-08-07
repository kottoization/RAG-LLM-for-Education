import importlib


def test_process_knowledge_refreshes_retriever(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class DummyRag:
        def __init__(self):
            self.retriever_calls = 0
        def ingest_paths(self, paths):
            pass
        def get_retriever(self, *args, **kwargs):
            self.retriever_calls += 1
            return f"retriever-{self.retriever_calls}"

    dummy = DummyRag()
    monkeypatch.setattr("tools.rag_service.get_rag_service", lambda: dummy)

    interface = importlib.reload(importlib.import_module("frontend_service.interface"))
    original = interface.retriever

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")
    file_obj = type("F", (), {"name": str(file_path)})

    # consume the generator to completion
    list(interface.process_knowledge([file_obj]))

    assert dummy.retriever_calls == 2
    assert interface.retriever != original
