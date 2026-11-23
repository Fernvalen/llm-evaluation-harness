# LLM Evaluation Harness

A Python-based evaluation harness for assessing Retrieval-Augmented Generation (RAG) outputs with structured, schema-validated metrics. It uses configurable LLM providers (local Ollama, OpenAI, or Google Gemini) to score responses and return explainable results as Pydantic models.

## Features

- **Pluggable LLM providers**
  - `LOCAL` (Ollama)
  - `OPENAI`
  - `GOOGLE` (Gemini)
- **Structured evaluation outputs** via Pydantic schemas
- **RAG quality metrics**
  - **Faithfulness**: checks whether response claims are grounded in retrieved context
  - **Relevancy**: checks whether the response addresses the user query
  - **Context Precision**: checks how much retrieved context is relevant vs. noisy
  - **Context Recall**: checks whether retrieved context covers ground-truth facts
- **Pytest-based quality gate tests** for metric behavior

## Project structure

```text
.
├── evaluators/
│   ├── engine.py        # RAGEvaluator and evaluation prompts/retry logic
│   └── schemas.py       # Pydantic metric schemas
├── providers/
│   ├── base.py          # Provider interface
│   ├── llm_factory.py   # Provider selection from environment
│   ├── ollama_provider.py
│   ├── openai_provider.py
│   └── google_provider.py
├── tests/
│   ├── conftest.py      # Pytest fixtures and model warm-up
│   └── test_*.py        # Metric quality-gate tests
├── .env.example
└── requirements.txt
```

## Requirements

- Python 3.10+ (recommended)
- Dependencies listed in [`requirements.txt`](requirements.txt)
- For local evaluation: a running [Ollama](https://ollama.com/) instance and a pulled model
- For cloud evaluation: valid OpenAI and/or Google API credentials

## Setup

1. **Create and activate a virtual environment** (optional but recommended)

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy [`.env.example`](.env.example) to `.env` and update values as needed.

   ```powershell
   Copy-Item .env.example .env
   ```

   Supported configuration (as read by the code):

   | Variable | Description | Default / Example |
   | --- | --- | --- |
   | `LLM_EXECUTION_MODE` | Provider mode: `LOCAL`, `OPENAI`, or `GOOGLE` | `LOCAL` |
   | `OLLAMA_BASE_URL` | Ollama server URL (local mode) | `http://localhost:11434` |
   | `OLLAMA_MODEL` | Ollama model name (local mode) | `llama3.2` |
   | `OPENAI_API_KEY` | OpenAI API key (`OPENAI` mode) | — |
   | `OPENAI_MODEL` | OpenAI model name (`OPENAI` mode) | `gpt-40` (code default) |
   | `GOOGLE_API_KEY` / `GEMINI_API_KEY`* | Google/Gemini API key (`GOOGLE` mode) | — |
   | `GOOGLE_MODEL` / `GEMINI_MODEL` | Gemini model name (`GOOGLE` mode) | `gemini-3.8-flash` (code default) |

   \* Check provider implementation/environment naming in your local `.env`: the factory and providers read specific variable names (e.g. `LLM_EXECUTION_MODE`, `OPENAI_MODEL`, `GOOGLE_MODEL`), while `.env.example` may contain differently named placeholders. Align your `.env` with the code/config you intend to use.

## Usage

Example using the evaluator API directly:

```python
from evaluators.engine import RAGEvaluator

evaluator = RAGEvaluator()

# Faithfulness: response vs retrieved context
faithfulness = evaluator.evaluate_faithfulness(
    query="What is the return policy for damaged items?",
    context="Damaged items can be returned within 30 days of purchase with proof of purchase for a full refund.",
    response="You can return damaged items within 30 days with proof of purchase for a full refund.",
)
print(faithfulness.score, faithfulness.reasoning, faithfulness.unsupported_claims)

# Relevancy: response vs query
relevancy = evaluator.evaluate_relevancy(
    query="What is the return policy for damaged items?",
    response="You can return damaged items within 30 days with proof of purchase.",
)
print(relevancy.score, relevancy.missing_aspects)

# Context precision: retrieved context vs query
precision = evaluator.evaluate_context_precision(
    query="What is the return policy for damaged items?",
    context="Damaged items can be returned within 30 days. Our stores are open from 9am to 5pm.",
)
print(precision.score, precision.irrelevant_chunks)

# Context recall: retrieved context vs ground truth
recall = evaluator.evaluate_context_recall(
    query="What is the return policy for damaged items?",
    context="Damaged items can be returned within 30 days of purchase with proof of purchase.",
    ground_truth="Damaged items can be returned within 30 days of purchase with proof of purchase for a full refund.",
)
print(recall.score, recall.unretrieved_facts)
```

Each metric method returns a Pydantic model containing:
- `score` (float, typically `0.0`–`1.0`)
- `reasoning` (step-by-step evaluation explanation)
- metric-specific fields (e.g. `unsupported_claims`, `missing_aspects`, `irrelevant_chunks`, `unretrieved_facts`)

## Running tests

The test suite uses `pytest` and includes LLM-backed quality-gate tests.

```powershell
pytest
```

Useful options:

```powershell
pytest -s          # show evaluator reasoning/print output
pytest -q          # quieter output
pytest -n auto     # run with xdist (if appropriate for your provider/rate limits)
```

> **Note:** Tests initialize the configured LLM provider and perform a warm-up call (`ping`) via a session-scoped fixture. Ensure your selected provider is reachable and properly configured before running the suite (e.g. Ollama running locally, or valid API keys/models for cloud providers).

## Metric schemas

Metric output contracts are defined in [`evaluators/schemas.py`](evaluators/schemas.py):

- `FaithfullnessMetric`
- `RelevancyMetric`
- `ContextPrecisionMetric`
- `ContextRecallMetric`

These schemas are used to request/validate structured LLM outputs during evaluation.

## Notes

- The evaluator includes a small retry wrapper around structured generation to handle transient/model parsing failures.
- Evaluation quality depends on the selected model/provider and prompt adherence to the expected schema.
- This repository currently provides the core evaluation engine, providers, schemas, and tests; extend it with your own datasets, pipelines, or reporting as needed.
