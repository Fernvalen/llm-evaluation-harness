class TestContextPrecisionQualityGates:

    def test_context_precision_detects_noisy_context(self, evaluator):
        query = "How do I reset my account password?"
        context = "To reset your password, click 'Forgot Password' on the login screen. Our company was founded in 1998 in Boston and has over 500 employees worldwide."

        metric = evaluator.evaluate_context_precision(query=query, context=context)
        assert metric.score < 1.0 or len(metric.irrelevant_chunks) > 0, "Failed to detect noise in retrieved context."