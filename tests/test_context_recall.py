import pytest

class TestContextRecallQualityGates:
    def test_context_recall_passes_when_all_facts_retrieved(self, evaluator):
        query = "What are the requirements for password resets?"
        ground_truth = "Password resets require an active email address and 2FA confirmation via SMS."
        context = "To reset your password, you need an active email address on file. You will also be prompted for 2FA confirmation sent via SMS."

        metric = evaluator.evaluate_context_recall(query=query, context=context, ground_truth=ground_truth) 

        print(f"\n[RECALL SCORE]: {metric.score}")
        print(f"\n[REASONING]: {metric.reasoning}")
        assert metric.score >= 0.85, f"Context recall gate failed! Unretrieved facts: {metric.unretrieved_facts}"

    def test_context_recall_fails_when_context_is_incomplete(self, evaluator):
        query = "What are the requirements for password resets?"
        ground_truth = "Password resets require an active email address and 2FA confirmation via SMS."
        # Context is missing the 2FA requirement
        context = "To reset your password, you only need an active email address on file."
        
        metric = evaluator.evaluate_context_recall(query=query, context=context, ground_truth=ground_truth)

        print(f"\n[FAIL CASE RECALL SCORE]: {metric.score}")
        print(f"[UNRETRIEVED FACTS]: {metric.unretrieved_facts}")
        assert metric.score < 1.0, "Evaluator failed to detect missing ground truth facts in context."
        assert len(metric.unretrieved_facts) > 0, "Expected unretrieved facts to contain missing 2FA requirement."
