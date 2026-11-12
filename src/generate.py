import re
import os
import configparser
from typing import Callable

from .retrieve import retrieve_embedding

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))

_SYSTEM_PROMPT = _config.get(
    'prompt', 'system',
    fallback=(
        "You are a helpful assistant that answers questions using only the "
        "information from the reference passage attached below. If the passage "
        "is irrelevant, let the user know you don't have enough information to answer."
    )
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generateResponse(userInput: str, llm: Callable[[str], str]) -> str:
    """
    Generate a RAG response for *userInput*.

    Parameters
    ----------
    userInput : str
        The user's question.
    llm : Callable[[str], str]
        Any callable that accepts a prompt string and returns a response string.
        The caller is responsible for constructing and injecting this — the
        library is model-agnostic and has no SDK dependencies of its own.

    Example
    -------
    >>> import google.genai as genai
    >>> client = genai.Client(api_key="...")
    >>> call_llm = lambda prompt: client.models.generate_content(
    ...     model="gemini-2.0-flash", contents=prompt).text
    >>> answer = generateResponse("Who is Heathcliff?", call_llm)
    """
    retrieved_docs = retrieve_embedding(userInput)
    if not retrieved_docs:
        return "I don't have enough information to answer that question."

    # Join chunks and strip non-printable control characters
    embedding_doc = "\n\n".join(retrieved_docs)
    clean_embedding_doc = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', embedding_doc)
    print("Retrieved context preview:", clean_embedding_doc[:200])

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Question: {userInput}\n\n"
        f"Reference Passage:\n{clean_embedding_doc}"
    )

    raw_response = llm(prompt)

    # Strip markdown formatting from the response
    clean_response = re.sub(r'\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|~~', '', raw_response)
    return clean_response
