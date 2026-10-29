from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel

# Generic type constrained to Pydantic BaseModel
T = TypeVar("T", bound=BaseModel)

# Enforces rule: class that inherits from BaseLLMProvider must implement exact methods
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Standard text generation interface"""
        pass

    @abstractmethod
    def generate_structure(self, prompt: str, schema: Type[T]) -> T:
        """Enforce Pydantic schema outputs for deterministic evaluation"""
        pass
