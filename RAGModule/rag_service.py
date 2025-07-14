import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
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

        # embeddings client
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # load or create vector store
        if os.listdir(self.persist_directory):
            self.docsearch = FAISS.load_local(
                self.persist_directory,
                embeddings=self.embeddings
            )
        else:
            # get docs from data/RAG_files
            loader = DirectoryLoader(
                "data/RAG_files",
                glob="**/*.*"
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)

            # FAISS creation and save to the dir
            self.docsearch = FAISS.from_documents(
                split_docs,
                embedding=self.embeddings
            )
            self.docsearch.save_local(self.persist_directory)

    def get_retriever(self, k: int = 5):
        """
        Returns a retriever to be used in pipelines/chains.
        """
        return self.docsearch.as_retriever(
            search_kwargs={"k": k}
        )
