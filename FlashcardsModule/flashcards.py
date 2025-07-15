import json
import os
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from langchain.chains import RetrievalQA
from RAGModule.rag import RAGHandler

class Flashcard:
    """
    Represents a single flashcard with a question and an answer.
    """
    def __init__(self, question: str, answer: str):
        self.question = question.strip()
        self.answer = answer.strip()

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer
        }

    @staticmethod
    def from_dict(data):
        return Flashcard(question=data["question"], answer=data["answer"])


class FlashcardSet:
    """
    Manages a set of flashcards: generation, review, and saving.
    """
    def __init__(self, topic: str, flashcards=None, retriever=None):
        self.topic = topic.strip()
        self.flashcards = flashcards if flashcards else []
        self.retriever = retriever  # optional RAG retriever

    def add_flashcard(self, flashcard: Flashcard):
        self.flashcards.append(flashcard)

    def generate_from_quiz_text(self, raw_text: str):
        """
        Parses quiz-style text and extracts flashcards from question blocks.
        """
        blocks = raw_text.strip().split("\n\n")
        for block in blocks:
            try:
                question_match = re.search(r"Question:\s*(.*)", block)
                correct_match = re.search(r"Correct Answer:\s*([a-d])", block)
                options = re.findall(r"[a-d]\)\s*(.*)", block)

                if question_match and correct_match and options:
                    idx = ord(correct_match.group(1).lower()) - ord('a')
                    answer = options[idx]
                    self.add_flashcard(Flashcard(question=question_match.group(1), answer=answer))
            except Exception as e:
                print(f"⚠️ Error parsing block: {e}")

    def generate_from_prompt(self, topic_prompt: str, language: str = "en", use_rag: bool = False):
        """
        Uses an LLM (optionally with RAG) to generate flashcards based on a topic prompt.
        When ``use_rag`` is True, additional context from your indexed documents is
        fetched and prepended to the prompt.
        """
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, verbose=True)

        def _fetch_context(inputs):
            rag = RAGHandler()
            rag.load_vectorstore()
            ctx = rag.get_context(inputs["topic_prompt"], k=3)
            return {**inputs, "context": ctx}

        def _skip_context(inputs):
            return {**inputs, "context": ""}

        branch = RunnableBranch(
            (lambda d: d.get("use_rag", False), RunnableLambda(_fetch_context)),
            RunnableLambda(_skip_context)
        )

        def _build_prompt(inputs):
            context = inputs["context"]
            topic = inputs["topic_prompt"]
            prompt = ""
            if context:
                prompt += context + "\n\n"
            prompt += (
                f"You are an expert educator preparing students for a rigorous test or exam.\n"
                f"Generate a high-quality, detailed list of flashcards for the topic: \"{topic}\".\n"
                f"The flashcards should include:\n"
                f"- definitions of core concepts\n"
                f"- names and explanations of key theorems or formulas\n"
                f"- concrete, technical facts that are often tested\n"
                f"- pay attention to the detailed domain knowledge needed by specialists at the level indicated by the user\n"
                f"Each flashcard must follow this format:\n"
                f"Q: [Clear, technical question]\n"
                f"A: [Precise, exam-focused answer]\n\n"
                f"Don't include explanations, examples, or anything besides flashcards.\n"
                f"Respond in {inputs['language']}."
            )
            return prompt

        chain = RunnableSequence(branch, RunnableLambda(_build_prompt) | llm)

        try:
            # if retriever provided, use RAG for richer context
            if self.retriever:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.retriever
                )
                raw_output = qa.run(topic_prompt)
            else:
                response = chain.invoke({"topic_prompt": topic_prompt, "language": language, "use_rag": use_rag})
                raw_output = response.content

            pairs = re.findall(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\Z)", raw_output, re.DOTALL)

            for q, a in pairs:
                self.add_flashcard(Flashcard(q.strip(), a.strip()))

            print(f"✅ Generated {len(self.flashcards)} flashcards from prompt.")
        except Exception as e:
            print(f"❌ Error generating flashcards from prompt: {e}")

    # def run_cli_review(self):
    #     """
    #     CLI for reviewing flashcards. Asks user for input and shows correct answer.
    #     """
    #     print(f"\n📚 Reviewing flashcards for topic: {self.topic}")
    #     for i, card in enumerate(self.flashcards, start=1):
    #         print(f"\n{i}. {card.question}")
    #         input("Your answer: ")
    #         print(f"✅ Correct answer: {card.answer}")

    def save_to_file(self, base_dir="data/flashcards/"):
        """
        Saves the flashcard set to a JSON file.
        """
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.topic}_flashcards_{timestamp}.json"
        path = os.path.join(base_dir, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict_list(), f, indent=4, ensure_ascii=False)
            print(f"💾 Flashcards saved to {path}")
        except Exception as e:
            print(f"❌ Failed to save flashcards: {e}")

    @staticmethod
    def load_from_file(path: str):
        """
        Loads a flashcard set from a JSON file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                topic = os.path.basename(path).split("_flashcards_")[0]
                cards = [Flashcard.from_dict(fc) for fc in data]
                return FlashcardSet(topic=topic, flashcards=cards)
        except Exception as e:
            print(f"❌ Failed to load flashcards: {e}")
            return None

    def to_dict_list(self):
        """
        Returns list of flashcards as list of dicts (e.g. for JSON API).
        """
        return [fc.to_dict() for fc in self.flashcards]
