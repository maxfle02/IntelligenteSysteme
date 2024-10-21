#!/usr/bin/env python3

import os
import tiktoken
import semchunk
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .vector_db import load_documents_from_directory
from dotenv import load_dotenv, find_dotenv

DB_PATH = "data/vectordb"
LOADED_DOCUMENTS_FILE = "data/loaded_documents.txt"

encoder = tiktoken.encoding_for_model("gpt-4")

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


def token_counter(text):
    return len(encoder.encode(text))


def token_counter(text):
    return len(encoder.encode(text))


def initialize_vectordatabase():
    """
    Initializes the vector database.
    """
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

    if not os.path.exists(LOADED_DOCUMENTS_FILE):
        with open(LOADED_DOCUMENTS_FILE, "w") as f:
            f.write("")

    try:
        if not os.path.exists(os.path.join(DB_PATH, "faiss.index")):
            documents = load_documents_from_directory()
            print(f"Debug: Loaded documents: {documents}")  # Debugging statement

            if not documents:
                print("Warning: No documents loaded.")  # Debugging statement
                return

            chunked_documents = []
            for doc in documents:
                # print(
                #     f"Debug: Processing document: {doc.metadata['source']}"
                # )  # Debugging statement
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                else:
                    content = doc
                chunks = semchunk.chunk(
                    content, chunk_size=512, token_counter=token_counter
                )
                if not chunks:
                    print(
                        f"Warning: No chunks created for document: {doc.metadata['source']}"
                    )  # Debugging statement
                chunked_documents.extend(chunks)
                # print(
                #     f"Debug: Chunks created for document: {chunks}"
                # )  # Debugging statement

            if not chunked_documents:
                raise RuntimeError("No chunks were created from the documents.")

            vectordb = FAISS.from_texts(
                chunked_documents,
                embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
            )
            vectordb.save_local(DB_PATH)

        else:
            print("Vector database already exists. Skipping initialization.")
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing the vector database: {str(e)}"
        )


def add_new_documents_to_vectordatabase():
    """
    Adds new documents to the vector database without rewriting the entire database.
    """
    vectordb = FAISS.load_local(
        DB_PATH,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    loaded_files = set()
    with open(LOADED_DOCUMENTS_FILE, "r") as f:
        loaded_files = set(f.read().splitlines())

    new_documents = load_documents_from_directory()
    new_chunks = []

    for doc in new_documents:
        if doc.metadata["source"] not in loaded_files:
            if hasattr(doc, "page_content"):
                content = doc.page_content
            else:
                content = doc
            chunks = semchunk.chunk(
                content, chunk_size=512, token_counter=token_counter
            )
            new_chunks.extend(chunks)
            loaded_files.add(doc.metadata["source"])

    if new_chunks:
        vectordb.add_texts(new_chunks)
        vectordb.save_local(DB_PATH)

        with open(LOADED_DOCUMENTS_FILE, "w") as f:
            f.write("\n".join(loaded_files))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector Database Initialization Script"
    )
    parser.add_argument(
        "--add-new-docs",
        action="store_true",
        help="Add new documents to the vector database",
    )

    args = parser.parse_args()

    if args.add_new_docs:
        add_new_documents_to_vectordatabase()
    else:
        initialize_vectordatabase()
