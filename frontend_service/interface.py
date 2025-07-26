import io
import os
import sys
from contextlib import redirect_stdout
import warnings

from dotenv import load_dotenv
from langchain_core._api import LangChainDeprecationWarning
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AgentModule import create_agent
from QuizModule import generate_quiz, generate_learning_plan_from_quiz
from LearningPlanModule import LearningPlan
from SummaryModule import StudySummaryGenerator
from FlashcardsModule import FlashcardSet
from CheatSheetModule import CheatSheetGenerator
from tools.language_handler import LanguageHandler

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)

warnings.filterwarnings(
    "ignore",
    message="fields may not start with an underscore",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file or export the variable."
    )

agent = create_agent()

CSS = """
* {
  font-family: 'Segoe UI', Tahoma, sans-serif;
}
#chatbot .message.user {
  background-color: #e6f3ff;
  border-radius: 8px;
}
#chatbot .message.bot {
  background-color: #f0f0f0;
  border-radius: 8px;
}
"""


def respond(message: str, history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    language = LanguageHandler.choose_or_detect(message)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = agent.invoke({"input": message, "language": language})["output"]
    history = history + [(message, result)]
    logs = buffer.getvalue()
    return history, logs


def run_quiz_interface(subject: str, use_rag: bool) -> str:
    """Run the CLI quiz generation with auto answers."""
    language = LanguageHandler.choose_or_detect(subject)
    buffer = io.StringIO()
    import builtins

    def _fake_input(prompt: str = ""):
        return "a"

    with redirect_stdout(buffer):
        original_input = builtins.input
        builtins.input = _fake_input
        try:
            generate_quiz(subject, language=language, use_rag=use_rag)
        finally:
            builtins.input = original_input
    return buffer.getvalue()


def run_learning_plan_interface(name: str, goals: str) -> str:
    """Generate a learning plan from custom goals."""
    language = LanguageHandler.choose_or_detect(goals)
    plan = LearningPlan(user_name=name, user_language=language)
    goals_list = [g.strip() for g in goals.split(";") if g.strip()]
    user_input = {"goals": goals_list}
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        plan.generate_plan_from_prompt(user_input)
        plan.display_plan()
        plan.save_to_file()
    return buffer.getvalue()


def run_flashcards_generate(topic: str, use_rag: bool) -> tuple[list[dict], str]:
    """Generate flashcards from a topic."""
    language = LanguageHandler.choose_or_detect(topic)
    flashcards = FlashcardSet(topic)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        flashcards.generate_from_prompt(topic_prompt=topic, language=language, use_rag=use_rag)
        flashcards.save_to_file()
    return flashcards.to_dict_list(), buffer.getvalue()


def run_flashcards_review(path: str) -> str:
    """Review flashcards from a saved file (auto answer)."""
    flashcards = FlashcardSet.load_from_file(path)
    if not flashcards:
        return "Failed to load flashcards."
    buffer = io.StringIO()
    import builtins

    def _fake_input(prompt: str = ""):
        return ""

    with redirect_stdout(buffer):
        original_input = builtins.input
        builtins.input = _fake_input
        try:
            flashcards.run_cli_review()
        finally:
            builtins.input = original_input
    return buffer.getvalue()


def run_summary_interface(topic: str, use_rag: bool) -> str:
    """Generate a detailed study summary."""
    language = LanguageHandler.choose_or_detect(topic)
    summarizer = StudySummaryGenerator()
    return summarizer.generate_summary(topic, language=language, use_rag=use_rag)


def run_cheatsheet_interface(topic: str, use_rag: bool) -> str:
    """Generate a cheat sheet."""
    language = LanguageHandler.choose_or_detect(topic)
    generator = CheatSheetGenerator()
    return generator.generate_cheatsheet(topic, language=language, use_rag=use_rag)


def build_interface() -> gr.Blocks:
    """Create the Gradio UI replicating the CLI menu."""
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EduGen", elem_id="title")

        with gr.Tabs():
            # Chat tab
            with gr.TabItem("Chat with the bot"):
                chatbot = gr.Chatbot(elem_id="chatbot")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your message and press enter...", container=False)
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                logs = gr.Textbox(label="Terminal output", lines=8)

                def clear_history():
                    return [], ""

                msg.submit(respond, [msg, chatbot], [chatbot, logs])
                send.click(respond, [msg, chatbot], [chatbot, logs])
                clear.click(clear_history, None, [chatbot, logs])

            # Quiz tab
            with gr.TabItem("Generate quiz"):
                quiz_subject = gr.Textbox(label="Subject")
                quiz_rag = gr.Checkbox(label="Use RAG", value=False)
                quiz_btn = gr.Button("Start Quiz")
                quiz_output = gr.Textbox(label="Quiz Output", lines=10)
                quiz_btn.click(run_quiz_interface, [quiz_subject, quiz_rag], quiz_output)

            # Learning plan tab
            with gr.TabItem("Learning plan"):
                plan_name = gr.Textbox(label="Your name")
                plan_goals = gr.Textbox(label="Learning goals (semicolon separated)")
                plan_btn = gr.Button("Generate Plan")
                plan_output = gr.Textbox(label="Plan Output", lines=10)
                plan_btn.click(run_learning_plan_interface, [plan_name, plan_goals], plan_output)

            # Flashcards tab
            with gr.TabItem("Flashcards"):
                with gr.Accordion("Generate flashcards", open=True):
                    fc_topic = gr.Textbox(label="Topic")
                    fc_rag = gr.Checkbox(label="Use RAG", value=False)
                    fc_gen_btn = gr.Button("Generate")
                    fc_cards = gr.JSON(label="Flashcards")
                    fc_logs = gr.Textbox(label="Logs", lines=4)
                    fc_gen_btn.click(run_flashcards_generate, [fc_topic, fc_rag], [fc_cards, fc_logs])

                with gr.Accordion("Review flashcards", open=False):
                    fc_path = gr.Textbox(label="Path to flashcards JSON")
                    fc_rev_btn = gr.Button("Review")
                    fc_review_out = gr.Textbox(label="Review Output", lines=10)
                    fc_rev_btn.click(run_flashcards_review, fc_path, fc_review_out)

            # Summary tab
            with gr.TabItem("Summary"):
                sum_topic = gr.Textbox(label="Topic or material")
                sum_rag = gr.Checkbox(label="Use RAG", value=False)
                sum_btn = gr.Button("Generate Summary")
                sum_output = gr.Textbox(label="Summary", lines=10)
                sum_btn.click(run_summary_interface, [sum_topic, sum_rag], sum_output)

            # Cheat sheet tab
            with gr.TabItem("Cheat sheet"):
                cs_topic = gr.Textbox(label="Topic or material")
                cs_rag = gr.Checkbox(label="Use RAG", value=False)
                cs_btn = gr.Button("Generate Cheat Sheet")
                cs_output = gr.Textbox(label="Cheat Sheet", lines=10)
                cs_btn.click(run_cheatsheet_interface, [cs_topic, cs_rag], cs_output)

    return demo


def launch_gradio() -> None:
    demo = build_interface()
    demo.queue()
    demo.launch()
