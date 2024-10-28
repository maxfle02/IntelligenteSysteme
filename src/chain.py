# config openAI
import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from .prompt import get_prompt
from .vector_db import get_retriever

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


## Auswahl korrekter chain!
# #https://python.langchain.com/docs/modules/chains/


def build_rag(retriever=None):
    """
    Builds a conversational RAG (Retrieval-Augmented Generation) chain.

    This function initializes and configures the components required for a conversational RAG chain,
    including the language model, retriever, and prompts. It creates a chain that can be used to
    generate responses based on a given chat history and user input.

    Returns:
        A conversational RAG chain that can be used to generate responses.

    Raises:
        Exception: If there is an error during initialization or configuration of the components.
    """
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    import logging

    logging.basicConfig(level=logging.ERROR)

    try:
        llm = ChatOpenAI(model="gpt-4o")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI: {str(e)}")
        raise

    try:
        if retriever is None:
            retriever = get_retriever()
    except Exception as e:
        logging.error(f"Failed to initialize the retriever: {str(e)}")
        raise

    qa_system_prompt = get_prompt()

    contextualize_q_system_prompt = "Beantworte die Eingabe direkt und füge keine zusätzlichen Informationen hinzu. Jede Eingabe endet mit '<|endoftext|>"

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input} <|endoftext|>"),
        ]
    )
    try:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
    except Exception as e:
        logging.error(f"Failed to create history aware retriever: {str(e)}")
        raise

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input} <|endoftext|>"),
        ]
    )

    try:
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    except Exception as e:
        logging.error(f"Failed to create QA chain: {str(e)}")
        raise

    try:
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
    except Exception as e:
        logging.error(f"Failed to create retrieval chain: {str(e)}")
        raise

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            logging.info(f"Creating new chat history for session {session_id}")
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain
