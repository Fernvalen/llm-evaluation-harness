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
    score: float = Field(
        description="Precision score from 0.0 (all noise/irrelevant context) to 1.0 (highly relevant context)."
    )