from pydantic import BaseModel, Field

class FaithfullnessMetric(BaseModel):
    """
    Evaluates whether the generated answer relies only on the retrieved context.
    Detects hallucinations or ungrounded claims
    """
    reasoning: str = Field(
        description="Step-by-step CoT reasoning evaluating if any claim is the response lacks support in the context."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of specific statements made in the response that are NOT supported by the context."
    )
    score: float = Field(
        description="Faithfulness score from 0.0 (completely hallucinated) to 1.0 (fully grounded in context)."
    )

class RelevancyMetric(BaseModel):
    """
    Evaluates wether the generated response directly and completely answers the user's query.
    Detects evasive, incomplete, or off-topic responses.
    """
    reasoning: str= Field(
        description="Step-by-step reasoning assessing if the response directly addresses every part of the user query."
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Lost of specific points from the user query that were ignored or missed in the response."
    )
    score: float = Field(
        description="Relevancy score from 0.0 (completely irrelevant) to 1.0 (perfectly addresses query)."     
    )

class ContextPrecisionMetric(BaseModel):
    """
    Evaluate the quality of the retrieved context relative to the user query.
    Detect noisy or irrelevant context chunks retrieved by vector search.
    """
    reasoning: str = Field(
        description="Analysis of the retrieved context chunks, identifying useful context vs noise."
    )
    irrelevant_chunks: list[str] = Field(
        default_factory=list,
        description="List of context sentences or chunks that add noise and do not help answer the query."
    )
    score: float = Field(
        description="Precision score from 0.0 (all noise/irrelevant context) to 1.0 (highly relevant context)."
    )

class ContextRecallMetric(BaseModel):
    """
    Evaluates wether the retrieved context contains ALL necessary information
    from the ground truth to answer the user query completely,
    """
    reasoning: str = Field(
        description="Step-by-sep analysis comparing ground truth statements against the retrieved context"
    )
    unretrieved_facts: list[str] = Field(
        default_factory=list,
        description="List of key facts present in te ground truth that were MISSING from retrieved context."
    )
    score: float = Field(
        description="Recall ratio score from 0.0 (zero ground truth facts retrieved) to 1.0 (all ground truth facts retrieved)."
    )
