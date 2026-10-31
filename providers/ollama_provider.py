import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

        # Connection pooling & socket reuse
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("http//", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"Connection": "close"})

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "system": system_prompt, "stream": False},
            timeout=60 # Explicit timeout to prevent hanging tests
        )
        return response.json().get("response", "")

    def generate_structure(self, prompt: str, schema):
        # Uses Ollama's structured JSON schema enforcement
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "format": schema.model_json_schema(),
                "stream": False
            },
            timeout=60 # Explicit timeout to prevent hanging tests
        )
        return schema.model_validate_json(response.json().get("response", ""))
    