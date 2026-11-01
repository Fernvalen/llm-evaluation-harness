class TestRelevancyQualityGate:

    def test_relevancy_passes_on_complete_amswer(self, evaluator):
        query = "What is your business hours and phone number?"
        response = "We are open Monday through Friday from 9 AM to 5 PM EST. You can reach us at 1-800-555-0199."

        metric = evaluator.evaluate_relevancy(query=query, response=response)
        assert metric.score >= 0.85, f"Relevnacy gate failed! Missing: {metric.missing_aspects}"