from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from tools.language_handler import LanguageHandler
from langchain.chains import RetrievalQA

# TODO: optimize with pipeline, quering, give more detailed contents, maybe more examples :  with ML prompt there are no examples of algorithms ect. 

class StudySummaryGenerator:
    """
    Generates a detailed study guide based on a topic – intended for learning, not just review.
    Ideal for exam preparation.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.5,retriever=None):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.retriever = retriever

        self.base_prompt = PromptTemplate.from_template(
            """
You are an expert university lecturer helping a student prepare for a difficult exam.

Your task is to create a **detailed, well-structured study guide** for the following topic:
"{input}"

This is not a cheat sheet. Instead, it should be a **multi-section, rich summary** that could span multiple pages.

Include:
- Clear and accurate definitions of core terms
- Detailed explanations of major concepts
- Examples for included concepts
- Theorems and laws, with explanation and usage
- Key formulas and symbols, written clearly and contextually
- Representative examples that help explain how the knowledge is applied
- Bullet lists or bold text to highlight what's most important
- Contextual usage: Where/why this knowledge is applied in real tasks/tests
- Where useful: diagrams, formulas, logical steps

Style:
- Use markdown-like formatting (titles, bullet points, code blocks)
- Clear separation of sections
- Friendly and slightly explanatory tone (like a good tutor)

IMPORTANT:
- Make it long enough to cover the topic as if preparing a student to pass an exam
- Avoid conversational tone – this should be structured content

Only output the content. No introductions or commentary.

Respond in {language}.
"""
        ) 

    def generate_summary(self, input_text: str, language: str = "en") -> str:
        """
        Generate a detailed study summary using the configured LLM and prompt.
        """
        lang = LanguageHandler.choose_or_detect(input_text) if language == "auto" else language

        if self.retriever:
            # Retrieval-augmented generation
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever
            )
            return qa.run(input_text)

        chain = self.base_prompt | self.llm
        response = chain.invoke({"input": input_text, "language": lang})
        return response.content
