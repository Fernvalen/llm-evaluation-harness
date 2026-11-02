import os
from dotenv import load_dotenv

# Automatically finds the .env file  and loads to environment variables
load_dotenv()

from .base import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider

def get_llm_provider() -> BaseLLMProvider:
    # Read environment setting; default to local Ollama execution
    execution_mode = os.getenv("LLM_EXECUTION_MODE", "LOCAL").upper()

    if execution_mode == "LOCAL":
        print("\n[PROVIDER] Ollama")
        return OllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2")
        )
    elif execution_mode == "OPENAI":
        return OpenAIProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-40")
        )
    elif execution_mode == "GOOGLE":
        return GoogleProvider(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")
        )
    else:
        raise ValueError(
            f"Unsupported execution mode: '{execution_mode}'. "
            f"Supported model: LOCAL, OPENAI, GOOGLE."
            )
    