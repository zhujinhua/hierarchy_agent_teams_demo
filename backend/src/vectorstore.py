### Build or Load Vector Index for RAG

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


def get_or_create_vectorstore(persist_dir: str, collection_name: str):
    """
    Check if Chroma vector store exists at specified directory.
    If exists, load and return it; if not, create new one with documents.

    Args:
        persist_dir: Directory path to store/load the vector store
        collection_name: Name of the Chroma collection

    Returns:
        Chroma vector store instance
        Retriever object for querying
    """
    embd = DashScopeEmbeddings(model="text-embedding-v4")
    # Check if vector store already exists
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print(f"Loading existing vector store from {persist_dir}")
        # Load existing vector store
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embd,
            persist_directory=persist_dir
        )
        return vectorstore, vectorstore.as_retriever()

    # If no existing store, create new one
    print(f"No existing vector store found. Creating new one at {persist_dir}")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Creating and populating vector store...")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=embd,
        persist_directory=persist_dir
    )
    vectorstore.persist()  # Save to disk
    print(f"Vector store created and saved to {persist_dir}")
    return vectorstore, vectorstore.as_retriever()


if __name__ == "__main__":
    vectorstore, retriever = get_or_create_vectorstore(
        persist_dir="../../data/chroma_db",
        collection_name="rag-chroma"
    )
