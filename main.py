from dotenv import load_dotenv
from QuizModule import generate_quiz, generate_learning_plan_from_quiz
from LearningPlanModule import LearningPlan
from SummaryModule import StudySummaryGenerator
from FlashcardsModule import FlashcardSet
from CheatSheetModule import CheatSheetGenerator
from AgentModule import create_agent
from AgentModule.edu_agent import run_agent
from frontend_service import launch_gradio
from tools.auto_answer import auto_answer
from tools.language_handler import LanguageHandler
from RAGModule import RAGHandler
import os
import warnings
from langchain_core._api import LangChainDeprecationWarning

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

warnings.filterwarnings(
    "ignore",
    message="fields may not start with an underscore",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Create a single agent instance for handling on-demand questions
_agent = create_agent()


def prompt_input(prompt: str) -> str:
    """Input wrapper that auto-runs the agent on questions."""
    user = input(prompt).strip()
    if auto_answer(user, _agent):
        return prompt_input(prompt)
    return user


def chat_with_bot():
    """Chat with the assistant. Optionally use RAG for document context."""
    try:
        use_rag = (
            prompt_input("Enrich answers with your documents? (y/N): ").strip().lower()
            == "y"
        )

        chat_history = []

        if use_rag:
            rag = RAGHandler()
            rag.load_vectorstore()
            print(
                "RAG mode enabled. You can now chat with the bot. Type 'exit' or 'q' to quit."
            )
        else:
            rag = None
            print("You can now chat with the bot. Type 'exit' or 'q' to quit.")

        while True:
            query = prompt_input("You: ")
            if query.lower() in ["exit", "q"]:
                break

            if use_rag and rag:
                # Pre-load vector store so the agent can query documents
                pass
            language = LanguageHandler.choose_or_detect(query)
            answer, used_fallback = run_agent(
                query, executor=_agent, return_details=True
            )
            answer = LanguageHandler.ensure_language(answer, language)
            if used_fallback:
                notice = LanguageHandler.ensure_language(
                    "Wiadomość generowana przez LLM, sprawdź jej poprawność",
                    language,
                )
                answer = f"{notice}\n{answer}"
            print(f"AI: {answer}")
            chat_history.append((query, answer))

        print("---- Message History ----")
        for q, a in chat_history:
            print(f"You: {q}")
            print(f"AI: {a}")

    except Exception as e:
        print(f"An error occurred while chatting with the bot: {e}")


def main_menu():
    """
    Main menu for the application.
    """
    # Initialize RAG handler and retriever (optional)
    try:
        rag = RAGHandler()
        retriever = rag.get_retriever(k=5)
    except Exception as e:
        print(f"[RAG Init Error] {e}")
        retriever = None

    while True:
        print("\nSelect an option:")
        print("0. Set preferred language")
        print("1. Chat with the bot (no articles required)")
        print("2. Generate a quiz")
        print("3. Create a personalized learning plan")
        print("4. Flashcards: Generate from prompt")
        print("5. Flashcards: Review from file")
        print("6. Generate TL;DR Summary")
        print("7. Generate Cheat Sheet")
        print("8. Exit")

        choice = prompt_input("Enter the number of your choice: ").strip()

        if choice == "0":
            print(
                "Enter your preferred language code (e.g. en, pl, de, fr) or 'auto' to detect automatically each time:"
            )
            lang = prompt_input("Language: ").strip()
            LanguageHandler.set_language(lang)
            print(f"Language set to: {lang}")

        elif choice == "1":
            chat_with_bot()

        elif choice == "2":
            subject = prompt_input("Enter the subject for the quiz: ")
            language = LanguageHandler.choose_or_detect(subject)
            # pass retriever to quiz (RAG-enabled if available)
            use_rag = (
                prompt_input("Use RAG to generate quiz topics? (y/N): ").strip().lower()
                == "y"
            )
            generate_quiz(
                subject, language=language, use_rag=use_rag, retriever=retriever
            )

        elif choice == "3":
            print("\nSelect an option:")
            print("1. Take a quiz to generate a learning plan")
            print("2. Input custom learning goals")
            sub_choice = prompt_input("Enter your choice: ").strip()

            if sub_choice == "1":
                subject = prompt_input("Enter the subject for the quiz: ")
                language = LanguageHandler.choose_or_detect(subject)
                use_rag = (
                    prompt_input("Use RAG to generate quiz topics? (y/N): ")
                    .strip()
                    .lower()
                    == "y"
                )
                quiz_results = generate_quiz(
                    subject, language=language, use_rag=use_rag, retriever=retriever
                )
                user_name = prompt_input("Enter your name: ")
                generate_learning_plan_from_quiz(user_name, quiz_results, language)
            elif sub_choice == "2":
                user_name = prompt_input("Enter your name: ")  # TODO: consider deleting
                goals_input = prompt_input(
                    "Enter your learning goals (comma-separated): "
                )
                language = LanguageHandler.choose_or_detect(goals_input)
                user_input = {
                    "goals": [goal.strip() for goal in goals_input.split(",")]
                }
                plan = LearningPlan(user_name=user_name, user_language=language)
                plan.generate_plan_from_prompt(user_input)
                plan.display_plan()
                plan.save_to_file()
            else:
                print("Invalid choice. Please try again.")

        elif choice == "4":
            topic = prompt_input("Enter a topic for flashcard generation: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to flashcards
            flashcards = FlashcardSet(topic, retriever=retriever)
            use_rag = (
                prompt_input("Enrich flashcards with your documents? (y/N): ")
                .strip()
                .lower()
                == "y"
            )
            flashcards.generate_from_prompt(
                topic_prompt=topic,
                language=language,
                use_rag=use_rag,
                retriever=retriever,
            )
            print(flashcards.to_dict_list())
            flashcards.save_to_file()

        elif choice == "5":
            path = prompt_input("Enter path to flashcard JSON file: ")
            flashcards = FlashcardSet.load_from_file(path)
            if flashcards:
                flashcards.run_cli_review()

        elif choice == "6":
            topic = prompt_input("Enter the topic or material for TL;DR summary: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to summary
            summarizer = StudySummaryGenerator(retriever=retriever)
            use_rag = (
                prompt_input("Enrich summary with your documents? (y/N): ")
                .strip()
                .lower()
                == "y"
            )
            summary = summarizer.generate_summary(
                topic, language=language, use_rag=use_rag, retriever=retriever
            )
            print("\n📘 Summary:\n")
            print(summary)

        elif choice == "7":
            topic = prompt_input("Enter the topic or material for the cheat sheet: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to cheat sheet generator
            generator = CheatSheetGenerator(retriever=retriever)
            use_rag = (
                prompt_input("Enrich cheat sheet with your documents? (y/N): ")
                .strip()
                .lower()
                == "y"
            )
            cheatsheet = generator.generate_cheatsheet(
                topic, language=language, use_rag=use_rag, retriever=retriever
            )
            print("\n📄 Cheat Sheet:\n")
            print(cheatsheet)

        elif choice in ("8", "q", "quit"):
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        main_menu()
    else:
        launch_gradio()
