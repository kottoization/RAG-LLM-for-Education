import QuizModule.quiz_operations as qo
from langchain.schema.runnable import Runnable


class FakeLLM(Runnable):
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt, config=None):
        class Msg:
            content = "Algebra\nGeometry"
        return Msg()

    def batch(self, prompts, config=None, **kwargs):
        return [type("Msg", (), {"content": "What is 2+2?\nCorrect Answer: a"}) for _ in prompts]


def test_prepare_quiz_questions(monkeypatch):
    monkeypatch.setattr(qo, "ChatOpenAI", FakeLLM)
    questions = qo.prepare_quiz_questions("math")
    assert questions == [
        {"topic": "Algebra", "question": "What is 2+2?", "correct": "a"},
        {"topic": "Geometry", "question": "What is 2+2?", "correct": "a"},
    ]


class EmptyLLM(Runnable):
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt, config=None):
        class Msg:
            content = ""
        return Msg()

    def batch(self, prompts, config=None, **kwargs):
        return []


def test_prepare_quiz_questions_no_topics(monkeypatch):
    monkeypatch.setattr(qo, "ChatOpenAI", EmptyLLM)
    assert qo.prepare_quiz_questions("history") == []
