from .vector_db import load_documents_to_vectordatabase


def ask_rag(rag_chain=None, query=None, session_id=None):
    """Run the chain with the given query and all pages
    Args:
        chain: The chain to run
        query: The query to run
    Returns:
        The result of the chain
    """

    if rag_chain is None:
        raise ValueError("rag_chain must not be None.")
    if query is None:
        raise ValueError("query must not be None.")
    if session_id is None:
        raise ValueError("session_id must not be None.")

    # result = chain.run(input_documents=additional_information, question=query)
    rag_result = rag_chain.invoke({"input": query, "session_id": session_id})
    result = rag_result["answer"]
    return result


def get_additional_information(query, k=5):
    """
    Retrieves additional information based on the given query.

    Parameters:
    query (str): The query string.
    k (int): The number of results to retrieve. Default is 5.

    Returns:
    list: A list of additional information retrieved based on the query.
    """
    vectordb = load_documents_to_vectordatabase()
    vectordb = vectordb.as_retriever()
    answer = vectordb.invoke(query, k=k)
    return answer
