import os
from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.schema import Document
from langchain_openai import ChatOpenAI

class RAGHandler:
    """
    Core RAG functionality:
      - ingest documents from folder
      - split into chunks
      - build/load Chroma vectorstore
      - semantic search and QA chains
    """
    def __init__(
        self,
        rag_files_path: str = "data/RAG_files",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        # 📂 Paths and basic setup
        self.rag_path = Path(rag_files_path)
        self.persist_dir = persist_directory

        # ✂️ Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 📈 Embedding model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # 🤖 LLM for QA
        self.llm = ChatOpenAI(model=llm_model, temperature=0.2)

        # 🔄 Placeholder for the vectorstore
        self.vectordb: Optional[Chroma] = None

    def ingest(self) -> List[Document]:
        """
        Load all .txt and .pdf files from rag_path into LangChain Documents.
        """
        docs: List[Document] = []

        # 📄 Load TXT files
        for txt_path in self.rag_path.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {txt_path}: {e}")

        # 📄 Load PDF files
        for pdf_path in self.rag_path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {pdf_path}: {e}")

        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split raw documents into smaller chunks.
        """
        return self.text_splitter.split_documents(docs)

    def build_vectorstore(
        self,
        docs: Optional[List[Document]] = None,
        persist: bool = True
    ) -> Chroma:
        """
        Build (or rebuild) the Chroma vectorstore from provided docs (or ingest folder).
        """
        if docs is None:
            docs = self.ingest()
        chunks = self.split(docs)

        # ⚡ Build the vectorstore
        vectordb = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir
        )

        # 💾 Persist to disk for future loads
        if persist:
            vectordb.persist()

        self.vectordb = vectordb
        return vectordb

    def load_vectorstore(self) -> Chroma:
        """
        Load an existing Chroma vectorstore from persist_directory.
        """
        if self.vectordb is None:
            self.vectordb = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
        return self.vectordb

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Return top-k document chunks relevant to the query.
        """
        db = self.load_vectorstore()
        retriever = db.as_retriever(search_kwargs={"k": k})  # 🔍
        return retriever.get_relevant_documents(query)

    def get_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve top-k relevant chunks and concatenate their content.
        """
        docs = self.semantic_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])  # 📚

    def answer(self, query: str, k: int = 5) -> str:
        """
        Perform a simple RetrievalQA: retrieve k docs and answer with LLM.
        """
        db = self.load_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": k})
        )  # 🤖
        return qa_chain.run(query)

    def chat(
        self,
        chat_history: List[dict],
        query: str,
        k: int = 3
    ) -> dict:
        """
        ConversationalRetrievalChain: maintain context + retrieve.
        """
        db = self.load_vectorstore()
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=db.as_retriever(search_kwargs={"k": k})
        )  # 🗣️
        return conv_chain({"question": query, "chat_history": chat_history})
