import requests
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "system": system_prompt, "stream": False} 
        )
        return response.json().get("response", "")

    def generate_structured(self, prompt: str, schema):
        # Uses Ollama's structured JSON schema enforcement
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": self.prompt,
                "format": schema.model_json_schema(),
                "stream": False
            }
        )
        return schema.parse_raw(response.json().get("response", ""))
    