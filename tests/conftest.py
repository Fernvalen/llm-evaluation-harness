import sys
from pathlib import Path
from dotenv import load_dotenv
import pytest

# Find project root path
TESTS_DIR = Path(__file__).parent.resolve()
ROOT_DIR = TESTS_DIR.parent.resolve()
ENV_PATH = ROOT_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Ensure project root is in Python path for clean package imports
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluators.engine import RAGEvaluator

@pytest.fixture(scope="session", autouse=True)
def warm_up_model():
    """
    SESSION FIXTURE:
    Pre-warms the configured LLM provider (Ollama, Gemini, OpenAI, etc.)
    once per test run to prevent cold-start timeout errors on initial evaluation calls.
    """
    evaluator = RAGEvaluator()
    evaluator.llm.generate(prompt="ping", system_prompt="respond with pong")

@pytest.fixture(scope="module")
def evaluator():
    """
    Shared RAGEvaluator instance supplied automatically to any test function requesting it.
    """
    return RAGEvaluator()