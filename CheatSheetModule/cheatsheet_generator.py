from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from RAGModule.rag import RAGHandler

# TODO: test and optimize

class CheatSheetGenerator:
    """
    Generates concise exam-style cheat sheets with only the most critical facts, formulas, and definitions.
    Ideal for rapid last-minute review. Use the Pareto principle.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.3,retriever=None):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.retriever = retriever

        self.prompt = PromptTemplate.from_template(
            """
{context}
You are an assistant that generates compact, high-quality cheat sheets to help students quickly review before exams.

Your task is to generate a **1-page cheat sheet** for the topic:
"{input}"

Include:
- Only the most essential definitions, formulas, key terms
- Structured layout with headers, bullet points, and highlights
- No detailed explanations – just what the student must memorize
- Formulas should be clearly presented
- Use markdown-style formatting (headers, bullet lists)

Written content should be helpful for someone who does not know the details and answer to given questions.
Focus on making the cheat sheet high quality and easy to use when faces a difficult question that one is not familiar with.
DO NOT include examples or commentary.
Only return the structured content.

Respond in this language only: {language} 
"""
        )

    def generate_cheatsheet(
        self,
        input_text: str,
        language: str = "en",
        use_rag: bool = False
    ) -> str:
        """
        Generate a cheat sheet; uses RAG if use_rag=True.

        Args:
            input_text: topic or material for which to generate the cheat sheet
            language: language code for the output
            use_rag: if True, fetch additional context from your documents

        Returns:
            Generated cheat sheet as string.
        """
        def _fetch_context(inputs):
            """Retrieve context from the provided retriever or a temporary RAG handler."""
            if self.retriever is not None:
                docs = self.retriever.get_relevant_documents(inputs["input"])
                ctx = "\n\n".join(doc.page_content for doc in docs)
            else:
                rag = RAGHandler()
                rag.load_vectorstore()
                ctx = rag.get_context(inputs["input"], k=3)
            return {"input": inputs["input"], "language": inputs["language"], "context": ctx}

        def _skip_context(inputs):
            return {"input": inputs["input"], "language": inputs["language"], "context": ""}

        branch = RunnableBranch(
            (lambda d: d.get("use_rag", False), RunnableLambda(_fetch_context)),
            RunnableLambda(_skip_context)
        )

        chain = RunnableSequence(branch, self.prompt | self.llm)

        response = chain.invoke({"input": input_text, "language": language, "use_rag": use_rag})
        return response.content
