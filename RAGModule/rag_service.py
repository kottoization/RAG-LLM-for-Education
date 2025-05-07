import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGService:
    """
    Service to ingest documents from folder, embed, index into FAISS,
    and provide a retriever for RAG use in other modules.
    """
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-ada-002",
        persist_directory: str = "data/vectorstore"
    ):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        if os.listdir(self.persist_directory):
            self.docsearch = FAISS.load_local(
                self.persist_directory,
                embeddings=self.embeddings
            )
        else:
            loader = DirectoryLoader(
                "data",
                glob="**/*.*"
            )  # <3
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)

            self.docsearch = FAISS.from_documents(
                split_docs,
                embedding=self.embeddings
            )
            self.docsearch.save_local(self.persist_directory)  # <3

    def get_retriever(self, k: int = 5):
        """
        Zwraca retriever do użycia w chainach.
        """
        return self.docsearch.as_retriever(
            search_kwargs={"k": k}
        )  # <3
