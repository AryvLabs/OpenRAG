from google import genai
from .retrieve import retrieve_embedding
import re
import os
import configparser

# Load config once at module level
_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
_SYSTEM_PROMPT = _config.get(
    'prompt', 'system',
    fallback="You are a helpful assistant that answers questions using only the information from the reference passage attached below. If the passage is irrelevant, let the user know you don't have enough information to answer."
)


def generateResponse(userInput):
    llmKey = os.getenv("GEMINI_API_KEY")
    if not llmKey:
        raise Exception("LLM API Key not present in Environment File")
    genAIProvider = genai.Client(api_key=llmKey)
    retrieved_docs = retrieve_embedding(userInput)
    if not retrieved_docs:
        return "I don't have enough information to answer that question."
    # Join multiple chunks and only strip non-printable/control characters
    embeddingDoc = "\n\n".join(retrieved_docs)
    cleanEmbeddingDoc = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', embeddingDoc)
    print("Retrieved context preview:", cleanEmbeddingDoc[:200])
    prompt = f"{_SYSTEM_PROMPT} Question:{userInput} Reference Passage:{cleanEmbeddingDoc}"
    genAIResponse = genAIProvider.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    raw_response = genAIResponse.text
    # Strip markdown formatting characters from the generated response
    clean_response = re.sub(r'\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|~~', '', raw_response)
    return clean_response
