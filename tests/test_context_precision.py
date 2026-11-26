import pytest
from evaluators.dataset import load_dataset_by_metric

precision_cases = load_dataset_by_metric("Precision")

class TestContextPrecisionQualityGates:

    @pytest.mark.parametrize("case", precision_cases, ids=lambda c: c["id"])
    def test_context_precision_gate(self, evaluator, case):
        query = case["query"]
        context = case["context"]
        expected_outcome = case.get("expected_outcome", "UNKNOWN")

        metric = evaluator.evaluate_context_precision(query=query, context=context)

        print(f"\n[{case['id']} - {case['topic']}] Score: {metric.score}")
        print(f"\n[REASONING]: {metric.reasoning}")

        if expected_outcome == "PASS":
            assert metric.score >= 0.85, (
                f"\nExpected 'PASS' but Precision score below threshold: {metric.score}"
                f"\n[IRRELEVANT CHUNKS]: {metric.irrelevant_chunks}"
            )
        else:
            assert metric.score < 0.85, (
                f"\nExpected 'FAIL' (Noisy Context) but Precision score above threshold: {metric.score}"
            )

