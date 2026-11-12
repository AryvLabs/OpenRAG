import os
import configparser
from typing import Callable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))

_PROVIDER = _config.get('llm', 'provider',    fallback='gemini').lower()
_MODEL    = _config.get('llm', 'model',       fallback='gemini-2.0-flash')
_KEY_ENV  = _config.get('llm', 'api_key_env', fallback='LLM_API_KEY')


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_llm() -> Callable[[str], str]:
    """
    Build and return an LLM callable based on config.ini [llm] settings.

    The returned callable accepts a prompt string and returns a response string.
    Provider-specific SDK imports are deferred to call time so that only the
    SDK for the configured provider needs to be installed.

    Supported providers (via config.ini):
        gemini, openai, anthropic

    For any other provider, construct the callable manually and pass it
    directly to generateResponse().
    """
    api_key = os.getenv(_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"LLM API key not found. Expected environment variable: '{_KEY_ENV}' "
            f"(set via 'api_key_env' in config.ini [llm] section)."
        )

    if _PROVIDER == 'gemini':
        from google import genai
        client = genai.Client(api_key=api_key)
        return lambda prompt: client.models.generate_content(
            model=_MODEL, contents=prompt
        ).text

    if _PROVIDER == 'openai':
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return lambda prompt: client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

    if _PROVIDER == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        return lambda prompt: client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ).content[0].text

    raise RuntimeError(
        f"Unsupported provider '{_PROVIDER}' in config.ini. "
        f"Supported values: gemini, openai, anthropic. "
        f"For other providers, construct your own callable and pass it to generateResponse() directly."
    )
