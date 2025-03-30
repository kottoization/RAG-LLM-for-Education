from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSequence


class StudySummaryGenerator:
    """
    Generates concise study summaries based on a topic or custom input.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        self.prompt = PromptTemplate.from_template(
            """
You are a helpful and highly knowledgeable assistant that specializes in creating focused, exam-oriented study summaries.

Your task is to generate a clear, structured TL;DR summary for the following topic or material:
"{input}"

The summary must:
- Cover only the most essential concepts, theorems, definitions, and facts
- Avoid any unnecessary elaboration or examples
- Use short, bullet-pointed or structured paragraphs
- Be useful for quick review before an exam

Respond ONLY with the summary. Do not include introductions or conclusions.
"""
        )

    def generate_summary(self, input_text: str) -> str:
        """
        Generates a summary string from user input.
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"input": input_text})
        return response.content
