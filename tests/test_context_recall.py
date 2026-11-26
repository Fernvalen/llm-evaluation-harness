import pytest
from evaluators.dataset import load_dataset_by_metric

recall_cases = load_dataset_by_metric("Recall")

class TestContextRecallQualityGates:

    @pytest.mark.parametrize("case", recall_cases, ids=lambda c: c["id"])
    def test_context_recall_gate(self, evaluator, case):
        query = case["query"]
        context = case["context"]
        ground_truth = case.get("grounded_truth", "")
        expected_outcome = case.get("expected_outcome", "UNKNOWN")

        metric = evaluator.evaluate_context_recall(query=query, context=context, ground_truth=ground_truth) 

        print(f"\n[{case['id']} - {case['topic']}] Score: {metric.score}")
        print(f"\n[REASONING]: {metric.reasoning}")

        if expected_outcome == "PASS":
            assert metric.score >= 0.85, (
                f"\nExpected 'PASS' but recall score below threshold: {metric.score}"
                f"\n[UNRETRIEVED FACTS]: {metric.unretrieved_facts}"
            )
        else:
            assert metric.score < 0.85,(
                f"\nExpected 'FAIL' (Missing Facts) but context recall above threshold: {metric.score}"
            )
