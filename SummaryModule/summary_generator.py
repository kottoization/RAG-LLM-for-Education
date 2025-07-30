from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from tools.language_handler import LanguageHandler
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from RAGModule.rag import RAGHandler

# Additional prompt for chain-of-density summarization

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

        self.dense_prompt = PromptTemplate.from_template(
            """
Article: {input}

You will generate increasingly concise, entity-dense summaries of the Article.

Repeat the following 2 steps 5 times:

Step 1. Identify 1-3 informative Entities (";" delimited) from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

A Missing Entity is:
- relevant to the main story
- specific yet concise (5 words or fewer)
- novel (not in the previous summary)
- faithful (present in the Article)
- anywhere (can be located anywhere in the Article)

Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Remember, use the exact same number of words for each summary.

Return only the final summary in {language}.
"""
        )

    def generate_summary(
        self,
        input_text: str,
        language: str = "en",
        use_rag: bool = False,
        retriever=None,
        dense: bool = False,
    ) -> str:
        """
        Generate a detailed study summary using the configured LLM and prompt.
        If ``use_rag`` is True, an external ``retriever`` can be supplied for
        context; otherwise a new :class:`RAGHandler` will be used.
        """
        lang = LanguageHandler.choose_or_detect(input_text) if language == "auto" else language

        retriever = retriever or self.retriever

        def _fetch_context(inputs):
            if retriever:
                """Retrieve additional context using RAG if a retriever is provided."""
                docs = retriever.get_relevant_documents(inputs["input"])
                ctx = "\n\n".join([doc.page_content for doc in docs])
            else:
                rag = RAGHandler()
                rag.load_vectorstore()
                ctx = rag.get_context(inputs["input"], k=3)
            inputs["input"] = f"{ctx}\n\n### Topic:\n{inputs['input']}"
            return inputs

        def _skip_context(inputs):
            return inputs

        branch = RunnableBranch(
            (lambda d: d.get("use_rag", False), RunnableLambda(_fetch_context)),
            RunnableLambda(_skip_context)
        )

        prompt = self.dense_prompt if dense else self.base_prompt

        chain = RunnableSequence(branch, prompt | self.llm)

        response = chain.invoke({"input": input_text, "language": lang, "use_rag": use_rag})
        return response.content
