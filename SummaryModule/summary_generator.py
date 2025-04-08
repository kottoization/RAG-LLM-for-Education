from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class StudySummaryGenerator:
    """
    Generates a detailed, exam-focused summary based on a topic.
    Applies the Pareto principle (80/20) to capture the most critical concepts.
    Include all of the most important bullet points, definitions, equations or concepts that are necessasy for a test.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.4):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        self.prompt = PromptTemplate.from_template(
            """
You are an expert educational assistant specialized in preparing students for university-level exams.

Your task is to generate a **structured , comprehensive, exam-oriented study summary** based on the topic:
"{input}"

Requirements:
- Apply the 80/20 principle – include only the 20% of content that covers 80% of what will be tested.
- For each item (definition, theorem, formula), briefly explain **why it's important** or **when it is used**.
- Cover:
    - core definitions and terminology **with context**
    - key theorems, concepts, and facts **with practical purpose**
    - essential formulas, equations, or syntax (for technical subjects) **with typical usage**
    - short examples **only if truly necessary for understanding**
- Format the summary in a structured way using bullet points or clear sections.
- Organized into sections (e.g. Definitions, Formulas, Theorems, Applications)
- Write in a professional but simple and memory-friendly tone (ideal for final review).
- Skip any irrelevant context, introductions, or general statements.
- Keep it clean, useful, and ideal for exam revision.

The result should serve as a **concise yet complete revision sheet** for an upcoming test or exam.

Respond ONLY with the structured summary.
"""
        )

    def generate_summary(self, input_text: str) -> str:
        """
        Generates a detailed exam-style study summary from user input.
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"input": input_text})
        return response.content
