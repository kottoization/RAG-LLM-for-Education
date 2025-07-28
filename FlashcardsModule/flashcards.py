import json
import os
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from langchain.chains import RetrievalQA
from RAGModule.rag import RAGHandler
from tools.auto_answer import auto_answer

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
        """Parse quiz-style text and extract flashcards from question blocks."""
        blocks = raw_text.strip().split("\n\n")
        for block in blocks:
            try:
                lines = block.strip().splitlines()
                if not lines:
                    continue
                # first line contains the question, potentially prefixed with
                # "Question:" or a numbering scheme like "1." or "1)"
                first_line = lines[0]
                first_line = re.sub(r"^(?:Question[:\s]*|\d+[.)]\s*)", "", first_line).strip()
                question = first_line
                correct_match = re.search(r"Correct Answer:\s*([a-d])", block, re.I)
                options = re.findall(r"[a-d]\)\s*(.*)", block)

                if question and correct_match and options:
                    idx = ord(correct_match.group(1).lower()) - ord("a")
                    answer = options[idx] if idx < len(options) else options[0]
                    self.add_flashcard(Flashcard(question=question, answer=answer))
            except Exception as e:
                print(f"⚠️ Error parsing block: {e}")

    def generate_from_prompt(self, topic_prompt: str, language: str = "en", use_rag: bool = False, retriever=None):
        """
        Uses an LLM (optionally with RAG) to generate flashcards based on a topic prompt.
        When ``use_rag`` is True, additional context from your indexed documents is
        fetched and prepended to the prompt. If a ``retriever`` was supplied and
        ``use_rag`` is ``True``, it is used via ``RetrievalQA`` for generation.
        """
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, verbose=True)
        retriever = retriever or self.retriever

        def _fetch_context(inputs):
            if retriever:
                docs = retriever.get_relevant_documents(inputs["topic_prompt"])
                ctx = "\n\n".join([doc.page_content for doc in docs])
            else:
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
            # Use any available retriever only when RAG is enabled
            if use_rag and retriever:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )
                raw_output = qa.run(topic_prompt)
            else:
                response = chain.invoke({
                    "topic_prompt": topic_prompt,
                    "language": language,
                    "use_rag": use_rag,
                })
                raw_output = response.content

            pairs = re.findall(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\Z)", raw_output, re.DOTALL)

            for q, a in pairs:
                self.add_flashcard(Flashcard(q.strip(), a.strip()))

            print(f"✅ Generated {len(self.flashcards)} flashcards from prompt.")
        except Exception as e:
            print(f"❌ Error generating flashcards from prompt: {e}")

    def run_cli_review(self):
        """Simple CLI loop for reviewing the flashcards."""
        print(f"\n📚 Reviewing flashcards for topic: {self.topic}")
        for i, card in enumerate(self.flashcards, start=1):
            print(f"\n{i}. {card.question}")
            while True:
                user = input("Your answer: ")
                if not auto_answer(user):
                    break
            print(f"✅ Correct answer: {card.answer}")

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
