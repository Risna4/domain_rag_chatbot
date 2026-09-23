import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vector_store import search_vector_store
from prompt import create_prompt


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm():
    """
    Create the Groq language model.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add your Groq API key to the .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=800
    )

    return llm


# --------------------------------------------------
# ANSWER QUESTION
# --------------------------------------------------

def answer_question(question):
    """
    Retrieve relevant chunks and generate
    a grounded answer.
    """

    # Retrieve top relevant chunks
    results = search_vector_store(
        question,
        top_k=4
    )

    # No sufficiently relevant information
    if not results:

        return {
            "answer":
                "I could not find this information "
                "in the uploaded documents.",
            "sources": []
        }

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        source = result[
            "metadata"
        ]["source"]

        page = result[
            "metadata"
        ]["page"]

        text = result["text"]

        context_parts.append(
            f"""
SOURCE {number}
Document: {source}
Page: {page}

Content:
{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------

    prompt = create_prompt(
        question,
        context
    )

    # --------------------------------------------------
    # CALL LLM
    # --------------------------------------------------

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    answer = response.content

    # --------------------------------------------------
    # COLLECT SOURCES
    # --------------------------------------------------

    sources = []

    for result in results:

        source = {
            "source":
                result["metadata"]["source"],

            "page":
                result["metadata"]["page"],

            "score":
                result["score"]
        }

        # Avoid duplicate document/page entries
        if source not in sources:

            sources.append(source)

    return {
        "answer": answer,
        "sources": sources
    }