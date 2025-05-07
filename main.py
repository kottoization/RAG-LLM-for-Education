from dotenv import load_dotenv
from QuizModule import generate_quiz, generate_learning_plan_from_quiz
from LearningPlanModule import LearningPlan
from SummaryModule import StudySummaryGenerator
from FlashcardsModule import FlashcardSet
from CheatSheetModule import CheatSheetGenerator
from tools.language_handler import LanguageHandler
from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from RAGModule import RAGService
import os

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

def chat_with_bot():
    """
    Allows the user to chat freely with the bot, maintaining a chat history.
    """
    try:
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, verbose=True)
        chat_history = []  # Store conversation history

        # Initial system message
        system_message = SystemMessage(content="You are a helpful AI assistant.")
        chat_history.append(system_message)

        print("You can now chat with the bot. Type 'exit' or 'q' to quit.")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "q"]:  # Allow 'exit' or 'q' to quit
                break

            chat_history.append(HumanMessage(content=query))  # Add user message
            response = model.invoke(chat_history)  # Get AI response
            print(f"AI: {response.content}")

            chat_history.append(AIMessage(content=response.content))  # Add AI message

        print("---- Message History ----")
        for msg in chat_history:
            print(msg.content)

    except Exception as e:
        print(f"An error occurred while chatting with the bot: {e}")

def main_menu():
    """
    Main menu for the application.
    """
    # Initialize RAGService and retriever (optional)
    try:
        rag_service = RAGService()
        retriever = rag_service.get_retriever(k=5)
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

        choice = input("Enter the number of your choice: ").strip()

        if choice == "0":
            print("Enter your preferred language code (e.g. en, pl, de, fr) or 'auto' to detect automatically each time:")
            lang = input("Language: ").strip()
            LanguageHandler.set_language(lang)
            print(f"Language set to: {lang}")

        elif choice == "1":
            chat_with_bot()

        elif choice == "2":
            subject = input("Enter the subject for the quiz: ")
            language = LanguageHandler.choose_or_detect(subject)
            # pass retriever to quiz (RAG-enabled if available)
            generate_quiz(subject, language=language, retriever=retriever)

        elif choice == "3":
            print("\nSelect an option:")
            print("1. Take a quiz to generate a learning plan")
            print("2. Input custom learning goals")
            sub_choice = input("Enter your choice: ").strip()

            if sub_choice == "1":
                subject = input("Enter the subject for the quiz: ")
                language = LanguageHandler.choose_or_detect(subject)
                quiz_results = generate_quiz(subject, language=language, retriever=retriever)
                user_name = input("Enter your name: ")
                generate_learning_plan_from_quiz(user_name, quiz_results, language)
            elif sub_choice == "2":
                user_name = input("Enter your name: ")  # TODO: consider deleting
                goals_input = input("Enter your learning goals (comma-separated): ")
                language = LanguageHandler.choose_or_detect(goals_input)
                user_input = {"goals": [goal.strip() for goal in goals_input.split(",")]}
                plan = LearningPlan(user_name=user_name, user_language=language)
                plan.generate_plan_from_prompt(user_input)
                plan.display_plan()
                plan.save_to_file()
            else:
                print("Invalid choice. Please try again.")

        elif choice == "4":
            topic = input("Enter a topic for flashcard generation: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to flashcards
            flashcards = FlashcardSet(topic, retriever=retriever)
            flashcards.generate_from_prompt(topic, language=language)
            print(flashcards.to_dict_list())
            flashcards.save_to_file()

        elif choice == "5":
            path = input("Enter path to flashcard JSON file: ")
            flashcards = FlashcardSet.load_from_file(path)
            if flashcards:
                flashcards.run_cli_review()

        elif choice == "6":
            topic = input("Enter the topic or material for TL;DR summary: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to summary
            summarizer = StudySummaryGenerator(retriever=retriever)
            summary = summarizer.generate_summary(topic, language=language)
            print("\n📘 Summary:\n")
            print(summary)

        elif choice == "7":
            topic = input("Enter the topic or material for the cheat sheet: ")
            language = LanguageHandler.choose_or_detect(topic)
            # pass retriever to cheat sheet generator
            generator = CheatSheetGenerator(retriever=retriever)
            cheatsheet = generator.generate_cheatsheet(topic, language=language)
            print("\n📄 Cheat Sheet:\n")
            print(cheatsheet)

        elif choice in ("8", "q", "quit"):
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
