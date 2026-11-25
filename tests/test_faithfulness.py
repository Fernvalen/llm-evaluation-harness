import pytest
from evaluators.dataset import load_dataset_by_metric

faithfulness_cases = load_dataset_by_metric("faithfulness")

class TestFaithfulnessQualityGates:

    @pytest.mark.parametrize("case", faithfulness_cases, ids=lambda c: c["id"])
    def test_faithfulness_gate(self, evaluator, case):
        """
        Parametrized quality gate for Faithfulnees.
        Evaluate grounding against context dynamically loaded from Golden Dataset.
        """
        query = case["query"]
        context = case["context"]
        response = case["response"]
        expected_outcome = case.get("expected_outcome", "UNKNOWN")

        # Execute evaluation via local Ollma or cloud API
        metric = evaluator.evaluate_faithfulness(query=query, context=context, response=response)

        # Print reasoning to terminal output for debugging (visible via 'pytest -s')
        print(f"\n[{case['id']} - {case['topic']}] Score: {metric.score}")
        print(f"[REASONING]: {metric.reasoning}")

        # Quality gate assertion
        if expected_outcome == "PASS":
            assert metric.score >= 0.85, (
                f"Expected 'PASS' but Faithfulness score dropped below threshold: {metric.score}."
                f"[UNSUPORTED CLAIMS]: {metric.unsupported_claims}"
            )
        else:
            assert metric.score < 0.85, (
                f"Expected 'FAIL' (Hallucination) but Faithfulness score passed: {metric.score}."
            )
    