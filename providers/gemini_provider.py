import os
from typing import Type, TypeVar
from pydantic import BaseModel
from google import genai
from google.genai import types
from .base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key must be provided or set in environment variables.")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard text generation using Gemini"""
        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system_prompt if system_prompt else None
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text or ""

    def generate_structure(self, prompt: str, schema: Type[T]) -> T:
        """Enforces native Pydantic schema outputs using Gemini Structured Outputs."""
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction="You are an automatedd evaluation judge. Output strict JSON matching the schema."
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        # Parse returned JSON text into our Pydantic model T
        return schema.model_validate_json(response.text)
    
