from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
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

Written content should be helpfull for someone who does not know the details and answer to given questions.
Focus on making the cheat sheet high quality and easy to use when faces a dificult question that one is not familiar with.
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
            use_rag: if True, retrieve relevant chunks and run RetrievalQA

        Returns:
            Generated cheat sheet as string.
        """
        if use_rag:
            # 🔍 Build/load vectorstore and retriever
            rag = RAGHandler()
            rag.load_vectorstore()
            retriever = rag.vectordb.as_retriever(search_kwargs={"k": 3})

            # 🤖 RetrievalQA chain
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )
            return qa.run(input_text)

        # 🔄 Fallback: standard prompt → LLM
        chain = self.prompt | self.llm
        response = chain.invoke({
            "input": input_text,
            "language": language
        })
        return response.content
