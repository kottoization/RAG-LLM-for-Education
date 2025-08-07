from langchain.schema.runnable import Runnable
from langchain_core.documents import Document

from SummaryModule.summary_generator import StudySummaryGenerator


def test_summary_generator_with_retriever(monkeypatch):
    class DummyRetriever:
        def invoke(self, query):
            assert query == "biology"
            return [Document(page_content="retrieved facts")]

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
        "SummaryModule.summary_generator.ChatOpenAI", lambda *a, **k: llm
    )

    gen = StudySummaryGenerator(retriever=DummyRetriever())
    result = gen.generate_summary("biology")

    assert "retrieved facts" in result
