import pytest

class TestRelevancyQualityGate:

    def test_relevancy_passes_on_complete_amswer(self, evaluator):
        query = "What is your business hours and phone number?"
        response = "We are open Monday through Friday from 9 AM to 5 PM EST. You can reach us at 1-800-555-0199."

        metric = evaluator.evaluate_relevancy(query=query, response=response)

        print(f"\n[RELEVANCY SCORE]: {metric.score}")
        print(f"\n[REASONING]: {metric.reasoning}")
        print(f"\n[MISSING ASPECTS]: {metric.missing_aspects}")

        assert metric.score >= 0.85, f"Relevancy gate failed! Missing: {metric.missing_aspects}"
