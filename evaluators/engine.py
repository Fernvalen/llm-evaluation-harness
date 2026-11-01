import time
from providers.llm_factory import get_llm_provider
from .schemas import FaithfullnessMetric, RelevancyMetric, ContextPrecisionMetric

class RAGEvaluator:
    def __init__(self):
        # Automatically resolved to Ollama or OpenAI based on .env configuration
        self.llm = get_llm_provider()

    def _execute_with_retry(self, prompt: str, schema, max_retries: int = 2):
        """ Retry loop for model parsing edge cases"""
        for attempt in range(max_retries + 1):
            try:
                return self.llm.generate_structure(prompt=prompt, schema=schema)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                time.sleep(1)

    def evaluate_faithfulness(self, query: str, context: str, response: str) -> FaithfullnessMetric:
        prompt_text = f"""
        You are a Quality Assurance Evaluator assessing RAG system outputs for faithfulness.
        
        INPUT DATA:
        - USER QUERY: {query}
        - RETRIEVED CONTEXT: {context}
        - GENERATED RESPONSE: {response}

        EVALUATION STEPS:
        Step 1. Extract every distinct factual claim or statement from the GENERATED RESPONSE.
        Step 2. Companre each claim against the RETRIEVED CONTEXT. 
        Step 3. In 'unsupported_claims', list ONLY claims that are NOT supported by or directly contradicts the context. (Ignore minor stylistic or synonym variations like 'goods' vs 'items').
        Step 4. Calculate the 'score'using this ratio formula:
                - Count total factual claims (N-total)
                - Count unsupported claims (N_unsupported)
                - If N_total == 0, score = 1.0
                - Otherwise, score = (N_total - N_unsupported) / N_total 
                - Output a float between 0.0 and 1.0
        5. Provide step-by-step CoT analysis in 'reasoning'
        """
        # Enforces output strictly into our FaithfulnessMetric schema
        return self._execute_with_retry(prompt=prompt_text, schema=FaithfullnessMetric)

    def evaluate_relevancy(self, query: str, response: str) ->RelevancyMetric:
        prompt_text = f"""
        You are a Quality Assurance Evaluator assessing whether AI response directly answers the user query.

        INPUT DATA:
        - USER QUERY: {query}
        - GENERATED RESPONSE: {response}
        
        EVALUATION STEPS:
        Step 1. Identify all main question components or requests in the USER QUERY.
        Step 2. Check if each component is answered in the GENERATED RESPONSE.
        Step 3. List any unanswered or ignored questions in the 'missing_aspects'.
        Step 4. Calculate 'score' from 0.0 (completely irrelevant) to 1.0 (fully addresses all parts of query).
        Step 5. Provide step-by-step CoT analysis in 'reasoning'
        """
        return self._execute_with_retry(prompt=prompt_text, schema=RelevancyMetric)

    def evaluate_context_precision(self, query:str, context: str) -> ContextPrecisionMetric:
        prompt_text = f"""
        You are a Quality Assurance Evaluator assessing the precision of vector retrieval context chunks.

        INPUT DATA:
        - USER QUERY: {query}
        - RETRIEVED CONTEXT: {context}

        EVALUATION STEPS:
        Step 1. Analyze the statements in the RETRIEVED CONTEXT relative to the USER QUERY.
        Step 2. Identify any context sentences or chunks that are irrelevant noise or off-topic.
        Step 3. List irrelevant statements in 'irrelevant_chunks'.
        Step 4. Calculate 'score' as: (Relevant Statements) / (Total Context Statements).
                - 1.0 = Highly focused context with zero noise
                - 0.0 = Entirely irrelevant noise.
        Step 5. Provide step-by-step CoT analysis in 'reasoning'. 
        """
        return self._execute_with_retry(prompt=prompt_text, schema=ContextPrecisionMetric)

    