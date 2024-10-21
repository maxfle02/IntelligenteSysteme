##  i need a function that will return a list of all the vectors in the database
# and a function to put data into the database
#

import os
import tiktoken
import semchunk
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .ingestion import load_documents_from_directory

DB_PATH = "data/vectordb"


def load_documents_to_vectordatabase():
    """
    Loads documents into the vector database.

    Returns:
        vectordb (FAISS): The loaded vector database.
    """
    vectordb_path = DB_PATH
    encoder = tiktoken.encoding_for_model("gpt-4")

    def token_counter(text):
        return len(encoder.encode(text))

    try:
        if os.path.exists(vectordb_path):
            vectordb = FAISS.load_local(
                DB_PATH,
                embeddings=OpenAIEmbeddings(),
                allow_dangerous_deserialization=True,
            )
        else:
            documents = load_documents_from_directory()
            chunked_documents = []
            for doc in documents:
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                else:
                    content = doc
                chunks = semchunk.chunk(
                    content, chunk_size=512, token_counter=token_counter
                )
                chunked_documents.extend(chunks)
            vectordb = FAISS.from_texts(chunked_documents, embedding=OpenAIEmbeddings())
            vectordb.save_local(DB_PATH)
    except Exception as e:
        # Handle exceptions such as file I/O errors, serialization issues, or any other unforeseen errors
        raise RuntimeError(
            f"An error occurred while processing the vector database: {str(e)}"
        )
    return vectordb


def get_vectors_from_vectordatabase():
    """
    Retrieves vectors from the vector database.

    Returns:
        dict: A dictionary containing the vectors stored in the vector database.
    """
    vectordb = load_documents_to_vectordatabase()
    vectors = vectordb.docstore._dict
    return vectors


def get_retriever(store=None):
    """
    Returns a retriever object for querying a vector database.

    Args:
        store (str): The name of the store to retrieve documents from. Defaults to None.

    Returns:
        retriever: A retriever object for querying the vector database.
    """
    db = load_documents_to_vectordatabase()
    retriever = db.as_retriever()
    return retriever
