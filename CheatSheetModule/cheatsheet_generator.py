from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# TODO: test and optimize

class CheatSheetGenerator:
    """
    Generates concise exam-style cheat sheets with only the most critical facts, formulas, and definitions.
    Ideal for rapid last-minute review. Use the Pareto principle.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        self.prompt = PromptTemplate.from_template(
            """
You are an assistant that generates compact, high-quality cheat sheets to help students quickly review before exams.

Your task is to generate a **1-page cheat sheet** for the topic:
"{input}"

Include:
- Only the most essential definitions, formulas, key terms
- Structured layout with headers, bullet points, and highlights
- No detailed explanations – just what the student must memorize
- Formulas should be clearly presented
- Use markdown-style formatting (headers, bullet lists)

DO NOT include examples or commentary.
Only return the structured content.

Respond in this language only: {language} ❤️
"""
        )

    def generate_cheatsheet(self, input_text: str, language: str = "en") -> str:
        chain = self.prompt | self.llm
        response = chain.invoke({
            "input": input_text,
            "language": language
        })
        return response.content
