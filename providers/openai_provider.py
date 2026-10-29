import os
from typing import Type, TypeVar
from pydantic import BaseModel
from openai import OpenAI
from .base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "gpt-40"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key must be provided or set in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard test generation using OpenAI's endpoint"""
        messages= []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].messages.content or ""

    def generate_structure(self, prompt: str, schema: type[T]) ->T:
        """Enforces native Pydantic schema outputs"""
        completion = self.client.beta.chat.completion.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an automated evaluation judge. Output strict JSON matching schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=schema,
            temperature=0.0
        )
        # Returns the automatically parsed Pydantic object
        return completion.choices[0].message.parsed