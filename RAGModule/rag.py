import os
from pathlib import Path
from typing import List, Optional

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

try:  # Prefer standalone package but fall back for compatibility
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - legacy support
    from langchain_community.vectorstores import Chroma
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
        # embedding_model: str = "text-embedding-3-large", TODO: use latest embedding model if needed, for now using the cheaper one
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        # 📂 Paths and basic setup
        self.rag_path = Path(rag_files_path)
        self.persist_dir = persist_directory

        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # ✂️ Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # 📈 Embedding model with lazy init
        # self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.embeddings: Optional[OpenAIEmbeddings] = None

        # 🤖 LLM for QA
        # self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        self.llm: Optional[ChatOpenAI] = None

        # 🔍 Cross-encoder reranker (lazy)
        self.reranker: Optional[
        ] = None
        self.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # 🔄 Placeholder for the vectorstore
        self.vectordb: Optional[Chroma] = None

    def _init_embeddings(self):
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

    def _init_llm(self):
        if self.llm is None:
            self.llm = ChatOpenAI(model=self.llm_model, temperature=0.2)

    def _init_reranker(self):
        if self.reranker is None:
            try:
                self.reranker = CrossEncoder(self.reranker_model)
            except Exception as e:
                print(f"❌ Error loading reranker: {e}")

    def ingest(self) -> List[Document]:
        """
        Load all .txt and .pdf files from rag_path into LangChain Documents.
        """
        docs: List[Document] = []

        # 📄 Load TXT files (search recursively)
        for txt_path in self.rag_path.rglob("*.txt"):
            try:
                loader = TextLoader(str(txt_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {txt_path}: {e}")

        # 📄 Load PDF files (search recursively)
        for pdf_path in self.rag_path.rglob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {pdf_path}: {e}")

        # 📄 Load CSV files (search recursively)
        for csv_path in self.rag_path.rglob("*.csv"):
            try:
                loader = CSVLoader(str(csv_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {csv_path}: {e}")

        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split raw documents into smaller chunks.
        """
        return self.text_splitter.split_documents(docs)

    def build_vectorstore(
        self, docs: Optional[List[Document]] = None, persist: bool = True
    ) -> Chroma:
        """
        Build (or rebuild) the Chroma vectorstore from provided docs (or ingest folder).
        """
        self._init_embeddings()
        if docs is None:
            docs = self.ingest()
        chunks = self.split(docs)

        # ⚡ Build the vectorstore
        vectordb = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_dir
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
        self._init_embeddings()
        if self.vectordb is None:
            db_path = Path(self.persist_dir) / "chroma.sqlite3"

            # Attempt to load existing DB if it exists
            if db_path.exists():
                self.vectordb = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir,
                )

                try:
                    # If no embeddings are stored, rebuild
                    if self.vectordb._collection.count() == 0:
                        self.vectordb = self.build_vectorstore()
                except Exception:
                    self.vectordb = self.build_vectorstore()
            else:
                self.vectordb = self.build_vectorstore()
        return self.vectordb

    def rerank_documents(
        self, docs: List[Document], query: str, top_k: int
    ) -> List[Document]:
        """Score and sort documents using the cross-encoder reranker."""
        self._init_reranker()
        if self.reranker is None:
            return docs[:top_k]
        pairs = [[query, doc.page_content] for doc in docs]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            print(f"❌ Error during reranking: {e}")
            return docs[:top_k]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def semantic_search(
        self, query: str, k: int = 3, use_rerank: bool = False
    ) -> List[Document]:
        """
        Return top-k document chunks relevant to the query. When ``use_rerank``
        is True, fetch more documents and sort them with a cross-encoder.
        """
        db = self.load_vectorstore()
        search_k = k * 3 if use_rerank else k
        retriever = db.as_retriever(search_kwargs={"k": search_k})  # 🔍
        docs = retriever.get_relevant_documents(query)
        if use_rerank:
            docs = self.rerank_documents(docs, query, k)
        return docs

    def get_context(self, query: str, k: int = 3, use_rerank: bool = False) -> str:
        """
        Retrieve top-k relevant chunks and concatenate their content.
        """
        docs = self.semantic_search(query, k=k, use_rerank=use_rerank)
        return "\n\n".join([doc.page_content for doc in docs])  # 📚

    def answer(self, query: str, k: int = 3, use_rerank: bool = False) -> str:
        """
        Perform a simple RetrievalQA: retrieve ``k`` docs and answer with LLM.
        When ``use_rerank`` is True, more documents are fetched and reranked
        before passing to the QA chain.
        """
        self._init_llm()
        db = self.load_vectorstore()
        if use_rerank:
            docs = self.semantic_search(query, k=k, use_rerank=True)

            class _StaticRetriever:
                def __init__(self, docs):
                    self.docs = docs

                def get_relevant_documents(self, _query):
                    return self.docs

                async def aget_relevant_documents(self, _query):
                    return self.docs

            retriever = _StaticRetriever(docs)
        else:
            retriever = db.as_retriever(search_kwargs={"k": k})

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
        )  # 🤖
        return qa_chain.run(query)

    def get_retriever(self, k: int = 5):
        """Return a retriever over the loaded vector store."""
        db = self.load_vectorstore()
        return db.as_retriever(search_kwargs={"k": k})

    def chat(self, chat_history: List[dict], query: str, k: int = 3) -> dict:
        """
        ConversationalRetrievalChain: maintain context + retrieve.
        """
        self._init_llm()
        db = self.load_vectorstore()
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, retriever=db.as_retriever(search_kwargs={"k": k})
        )  # 🗣️
        return conv_chain({"question": query, "chat_history": chat_history})
