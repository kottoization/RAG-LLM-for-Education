import FlashcardsModule.flashcards as fc
from langchain.schema.runnable import Runnable


class FakeLLM(Runnable):
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt, config=None):
        class Msg:
            content = (
                "Q: Capital of France?\nA: Paris\n"
                "Q: 2+2?\nA: 4"
            )
        return Msg()

    def batch(self, prompts, config=None, **kwargs):
        return [self.invoke(p) for p in prompts]


def test_generate_from_prompt(monkeypatch):
    monkeypatch.setattr(fc, "ChatOpenAI", FakeLLM)
    flashcards = fc.FlashcardSet("geo")
    flashcards.generate_from_prompt("geography")
    assert flashcards.flashcards[0].question == "Capital of France?"
    assert flashcards.flashcards[0].answer == "Paris"
    assert len(flashcards.flashcards) == 2


def test_generate_from_quiz_text():
    raw = (
        "Question: What is 2+2?\n"
        "a) 3\n"
        "b) 4\n"
        "c) 5\n"
        "d) 22\n"
        "Correct Answer: b\n\n"
        "Question: Capital of Italy?\n"
        "a) Rome\n"
        "b) Madrid\n"
        "c) Paris\n"
        "d) Berlin\n"
        "Correct Answer: a"
    )
    fs = fc.FlashcardSet("math")
    fs.generate_from_quiz_text(raw)
    assert len(fs.flashcards) == 2
    assert fs.flashcards[0].question == "What is 2+2?"
    assert fs.flashcards[0].answer == "4"
