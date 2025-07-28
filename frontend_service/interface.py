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
from QuizModule import generate_quiz, generate_learning_plan_from_quiz, prepare_quiz_questions
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


def respond(message: str, history: list[tuple[str, str]], lang_choice: str) -> tuple[list[tuple[str, str]], str]:
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(message)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = agent.invoke({"input": message, "language": language})["output"]
        result = LanguageHandler.ensure_language(result, language)
    history = history + [(message, result)]
    logs = buffer.getvalue()
    return history, logs


def _format_question(q: dict) -> str:
    """Return formatted question text with options on separate lines."""
    import re

    text = q["question"]
    # ensure each answer choice appears on its own line
    text = re.sub(r"\s*([abcd]\))", r"\n\1", text, flags=re.I)
    text = text.strip()
    return f"**{q['topic']}**\n\n{text}"


def start_quiz(subject: str, use_rag: bool, lang_choice: str) -> tuple[str, dict, str]:
    """Generate quiz questions and return the first one with state."""
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(subject)
    questions = prepare_quiz_questions(subject, language=language, use_rag=use_rag)
    if not questions:
        return "Failed to generate quiz.", {}, ""
    state = {
        "questions": questions,
        "index": 0,
        "scores": {},
        "correct_total": 0,
        "language": language,
    }
    first_q = _format_question(questions[0])
    return first_q, state, ""


def answer_quiz(choice: str, state: dict) -> tuple[str, dict, str]:
    """Process an answer button click and return next question or results."""
    if not state or state.get("index") is None:
        return "Quiz not started.", state, ""

    idx = state["index"]
    questions = state["questions"]
    if idx >= len(questions):
        return "", state, _compile_results(state)

    current = questions[idx]
    topic = current["topic"]
    correct = current["correct"]
    scores = state.setdefault("scores", {}).setdefault(topic, [0, 0])
    scores[1] += 1
    if choice.lower() == correct or correct == "?":
        scores[0] += 1
        state["correct_total"] += 1

    state["index"] += 1
    if state["index"] >= len(questions):
        return "", state, _compile_results(state)
    next_q = _format_question(questions[state["index"]])
    return next_q, state, ""


def _compile_results(state: dict) -> str:
    lines = []
    total_questions = 0
    total_correct = state.get("correct_total", 0)
    for topic, (corr, tot) in state.get("scores", {}).items():
        perc = (corr / tot) * 100 if tot else 0
        lines.append(f"{topic}: {corr}/{tot} ({perc:.2f}%)")
        total_questions += tot
    if lines:
        overall = sum((corr / tot) * 100 if tot else 0 for corr, tot in state["scores"].values())
        overall /= len(state["scores"])
        lines.append(f"\nOverall Score: {total_correct}/{total_questions} ({overall:.2f}%)")
    result = "\n".join(lines)
    lang = state.get("language", "auto")
    return LanguageHandler.ensure_language(result, lang)


def run_learning_plan_interface(name: str, goals: str, lang_choice: str) -> str:
    """Generate a learning plan from custom goals."""
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(goals)
    plan = LearningPlan(user_name=name, user_language=language)
    goals_list = [g.strip() for g in goals.split(";") if g.strip()]
    user_input = {"goals": goals_list}
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        plan.generate_plan_from_prompt(user_input)
        plan.display_plan()
        plan.save_to_file()
    return buffer.getvalue()


def run_flashcards_generate(topic: str, use_rag: bool, lang_choice: str) -> tuple[list[dict], str]:
    """Generate flashcards from a topic."""
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(topic)
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


def run_summary_interface(topic: str, use_rag: bool, lang_choice: str) -> str:
    """Generate a detailed study summary."""
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(topic)
    summarizer = StudySummaryGenerator()
    return summarizer.generate_summary(topic, language=language, use_rag=use_rag)


def run_cheatsheet_interface(topic: str, use_rag: bool, lang_choice: str) -> str:
    """Generate a cheat sheet."""
    code = LanguageHandler.code_from_display(lang_choice)
    language = code if code != "auto" else LanguageHandler.choose_or_detect(topic)
    generator = CheatSheetGenerator()
    return generator.generate_cheatsheet(topic, language=language, use_rag=use_rag)


def build_interface() -> gr.Blocks:
    """Create the Gradio UI replicating the CLI menu."""
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EduGen", elem_id="title")
        lang_select = gr.Dropdown(
            choices=LanguageHandler.dropdown_choices(),
            value=LanguageHandler.dropdown_choices()[0],
            label="Language"
        )

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

                msg.submit(respond, [msg, chatbot, lang_select], [chatbot, logs])
                send.click(respond, [msg, chatbot, lang_select], [chatbot, logs])
                clear.click(clear_history, None, [chatbot, logs])

            # Quiz tab
            with gr.TabItem("Generate quiz"):
                quiz_subject = gr.Textbox(label="Subject")
                quiz_rag = gr.Checkbox(label="Use RAG", value=False)
                start_btn = gr.Button("Start Quiz")
                quiz_question = gr.Markdown()
                with gr.Row():
                    btn_a = gr.Button("A")
                    btn_b = gr.Button("B")
                    btn_c = gr.Button("C")
                    btn_d = gr.Button("D")
                quiz_result = gr.Markdown()
                quiz_state = gr.State()

                start_btn.click(start_quiz, [quiz_subject, quiz_rag, lang_select], [quiz_question, quiz_state, quiz_result])
                btn_a.click(lambda st: answer_quiz("a", st), quiz_state, [quiz_question, quiz_state, quiz_result])
                btn_b.click(lambda st: answer_quiz("b", st), quiz_state, [quiz_question, quiz_state, quiz_result])
                btn_c.click(lambda st: answer_quiz("c", st), quiz_state, [quiz_question, quiz_state, quiz_result])
                btn_d.click(lambda st: answer_quiz("d", st), quiz_state, [quiz_question, quiz_state, quiz_result])

            # Learning plan tab
            with gr.TabItem("Learning plan"):
                plan_name = gr.Textbox(label="Your name")
                plan_goals = gr.Textbox(label="Learning goals (semicolon separated)")
                plan_btn = gr.Button("Generate Plan")
                plan_output = gr.Textbox(label="Plan Output", lines=10)
                plan_btn.click(run_learning_plan_interface, [plan_name, plan_goals, lang_select], plan_output)

            # Flashcards tab
            with gr.TabItem("Flashcards"):
                with gr.Accordion("Generate flashcards", open=True):
                    fc_topic = gr.Textbox(label="Topic")
                    fc_rag = gr.Checkbox(label="Use RAG", value=False)
                    fc_gen_btn = gr.Button("Generate")
                    fc_cards = gr.JSON(label="Flashcards")
                    fc_logs = gr.Textbox(label="Logs", lines=4)
                    fc_gen_btn.click(run_flashcards_generate, [fc_topic, fc_rag, lang_select], [fc_cards, fc_logs])

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
                sum_btn.click(run_summary_interface, [sum_topic, sum_rag, lang_select], sum_output)

            # Cheat sheet tab
            with gr.TabItem("Cheat sheet"):
                cs_topic = gr.Textbox(label="Topic or material")
                cs_rag = gr.Checkbox(label="Use RAG", value=False)
                cs_btn = gr.Button("Generate Cheat Sheet")
                cs_output = gr.Textbox(label="Cheat Sheet", lines=10)
                cs_btn.click(run_cheatsheet_interface, [cs_topic, cs_rag, lang_select], cs_output)

    return demo


def launch_gradio() -> None:
    demo = build_interface()
    demo.queue()
    demo.launch()
