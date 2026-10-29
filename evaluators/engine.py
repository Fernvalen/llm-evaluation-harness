from providers.llm_factory import get_llm_provider
from .schemas import FaithfullnessMetric, RelevancyMetric

class RAGEvaluator:
    def __init__(self):
        # Automatically resolved to Ollama or OpenAI based on .env configuration
        self.llm = get_llm_provider()

    def evaluate_faithfulness(self, query: str, context: str, response: str) -> FaithfullnessMetric:
        prompt = f"""
        Analyze the following RAG system for faithfulness:

        USER QUERY: {query}
        RETRIEVED CONTEXT: {context}
        GENERATED RESPONSE: {response}

        Task: Identify if the generated response contains any information that is NOT present in the retrieved context.
        """
        # Enforces output strictly into our FaithfulnessMetric schema
        return self.llm.generate_structured(prompt=prompt, schema=FaithfullnessMetric)
    