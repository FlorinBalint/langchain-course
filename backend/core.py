import os

from typing import Any, Dict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage

from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectors = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)

model = init_chat_model("gpt-5.2", model_provider = "openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """ Retrieves relevant documentation to help answer user queries about Langchain"""

    retrieved_docs = vectors.as_retriever().invoke(query, k=4)

    serialized = "\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\n Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized and the retrieved docs
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query from retrieved documentation.

    Args:
        query (str): The user's question

    Returns:
        Dictionary containing:
            - the answer: The generated answers
            - context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Do not include source citations or URLs in your answer. Sources are shown separately in the UI. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model=model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages" : messages})

    # Extract the answer from last AI message
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}

if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)
