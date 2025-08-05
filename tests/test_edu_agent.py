import pytest
import AgentModule.edu_agent as ea
import tools.language_handler as lh


class DummyExecutor:
    def __init__(self, output):
        self.output = output

    def invoke(self, inputs):
        return {"output": self.output}


class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        class Msg:
            content = "Fallback answer"
        return Msg()


@pytest.fixture
def patched_agent(monkeypatch):
    monkeypatch.setattr(lh.LanguageHandler, "choose_or_detect", lambda text: "en")
    monkeypatch.setattr(lh.LanguageHandler, "ensure_language", lambda text, lang: text)
    monkeypatch.setattr(ea, "ChatOpenAI", FakeLLM)
    return lambda output: DummyExecutor(output)


def test_run_agent_no_fallback(patched_agent):
    executor = patched_agent("The capital is Warsaw")
    output, used_fallback = ea.run_agent("Question", executor=executor, return_details=True)
    assert output == "The capital is Warsaw"
    assert used_fallback is False


def test_run_agent_with_fallback(patched_agent):
    executor = patched_agent("error: something broke")
    output, used_fallback = ea.run_agent("Question", executor=executor, return_details=True)
    assert output == "Fallback answer"
    assert used_fallback is True
