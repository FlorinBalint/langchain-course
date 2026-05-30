from typing import Dict, Any

from graph.chains.retriever_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANCE TO QUESTION---")
            filtered_documents.append(document)
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT TO QUESTION---")
            web_search = True
            continue

    return {"documents": filtered_documents, "question": question, "web_search": web_search}