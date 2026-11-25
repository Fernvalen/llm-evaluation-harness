import pytest
from evaluators.dataset import load_dataset_by_metric

relevancy_cases = load_dataset_by_metric("Relevancy")

class TestRelevancyQualityGate:

    @pytest.mark.parametrize("case", relevancy_cases, ids=lambda c: c["id"])
    def test_relevancy_passes_on_complete_amswer(self, evaluator, case):
        query = case["query"]
        response = case["response"]
        expected_outcome = case.get("expected_outcome", "UNKNOWN")

        metric = evaluator.evaluate_relevancy(query=query, response=response)

        print(f"\n[{case['id']} - {case['topic']}] Score: {metric.score}")
        print(f"\n[REASONING]: {metric.reasoning}")

        if expected_outcome == "PASS":
            assert metric.score >= 0.85, (
                f"Relevancy gate 'FAIL'."
                f"[MISSING ASPECTS]: {metric.missing_aspects}"
            )
        else:
            assert metric.score < 0.85, (
                f"Expected failure on evasive/irrelevant response but gate passed"
            )
