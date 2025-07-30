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
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.3, retriever=None):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.retriever = retriever

        # Based on the RStudio cheatsheet guidelines which suggest designing
        # materials in a clear three or four column layout for readability
        # (see https://github.com/rstudio/cheatsheets/blob/main/.github/CONTRIBUTING.md)
        self.prompt = PromptTemplate.from_template(
            """
{context}
You are an assistant that generates compact, exam-focused cheat sheets.

Create a concise **one page** cheat sheet for the topic:
"{input}"

Structure the sheet using these sections:
### Key Terms
- short bullet points (max ~12 words)

### Must-Know Formulas
- clear formulas or equations only

### Quick Facts
- bite-size facts useful for revision

Guidelines:
- Keep bullet lists extremely concise
- Use markdown headers and bullet lists
- Avoid long paragraphs, examples or commentary
- No images – text only for accessibility

Respond only in {language}.
"""
        )

    def generate_cheatsheet(
        self,
        input_text: str,
        language: str = "en",
        use_rag: bool = False,
        retriever=None
    ) -> str:
        """
        Generate a cheat sheet; uses RAG if use_rag=True.

        Args:
            input_text: topic or material for which to generate the cheat sheet
            language: language code for the output
            use_rag: if True, fetch additional context from your documents
            retriever: optional external retriever to supply that context

        Returns:
            Generated cheat sheet as string.
        """
        retriever = retriever or self.retriever

        def _fetch_context(inputs):
            if retriever:
                docs = retriever.get_relevant_documents(inputs["input"])
                ctx = "\n\n".join([doc.page_content for doc in docs])

            else:
                rag = RAGHandler()
                rag.load_vectorstore()
                ctx = rag.get_context(inputs["input"], k=3)
            return {
                "input": inputs["input"],
                "language": inputs["language"],
                "context": ctx,
            }


        def _skip_context(inputs):
            return {"input": inputs["input"], "language": inputs["language"], "context": ""}

        branch = RunnableBranch(
            (lambda d: d.get("use_rag", False), RunnableLambda(_fetch_context)),
            RunnableLambda(_skip_context)
        )

        chain = RunnableSequence(branch, self.prompt | self.llm)

        response = chain.invoke({"input": input_text, "language": language, "use_rag": use_rag})
        return response.content
