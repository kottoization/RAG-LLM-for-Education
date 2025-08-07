from langchain.schema.runnable import Runnable
from langchain_core.documents import Document

from CheatSheetModule.cheatsheet_generator import CheatSheetGenerator


def test_cheatsheet_generator_with_retriever(monkeypatch):
    class DummyRetriever:
        def invoke(self, query):
            assert query == "physics"
            return [Document(page_content="retrieved context")]

    class CapturingLLM(Runnable):
        def __init__(self, *args, **kwargs):
            self.last_prompt = None

        def invoke(self, prompt, config=None):
            self.last_prompt = prompt.text if hasattr(prompt, "text") else prompt
            class Msg:
                content = self.last_prompt
            return Msg()

    llm = CapturingLLM()
    monkeypatch.setattr(
        "CheatSheetModule.cheatsheet_generator.ChatOpenAI", lambda *a, **k: llm
    )

    generator = CheatSheetGenerator(retriever=DummyRetriever())
    result = generator.generate_cheatsheet("physics")

    assert "retrieved context" in result
