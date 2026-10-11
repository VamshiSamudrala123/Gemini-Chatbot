import unittest

from diagnostics import classify_failure, provider_operation


class SDKError(Exception):
    def __init__(self, code, message="private key, request URL, and question"):
        self.code = code
        super().__init__(message)


class DiagnosticTests(unittest.TestCase):
    def test_wrapped_sdk_quota_code_is_reported_without_private_data(self):
        try:
            with provider_operation("embedding"):
                raise SDKError(429)
        except Exception as exc:
            failure = classify_failure(exc)
        self.assertEqual(failure.status, 429)
        self.assertEqual(failure.stage, "embedding")
        self.assertIn("HTTP 429", failure.message)
        self.assertIn("zero quota", failure.message)
        self.assertNotIn("private key", failure.message)

    def test_http_categories_have_specific_fixed_messages(self):
        for code, expected in [
            (400, "parameters"), (401, "authenticate"), (402, "billing"),
            (403, "denied access"), (404, "model or resource"),
            (408, "timed out"), (500, "Google's service"), (503, "Google's service"),
        ]:
            with self.subTest(code=code):
                failure = classify_failure(SDKError(code))
                self.assertIn(expected, failure.message)
                self.assertNotIn("private key", failure.message)

    def test_invalid_key_marker_is_safe_and_actionable(self):
        failure = classify_failure(SDKError(400, "API_KEY_INVALID private-secret"))
        self.assertIn("rejected the API key", failure.message)
        self.assertNotIn("private-secret", failure.message)

    def test_free_tier_eligibility_is_distinct_from_quota(self):
        failure = classify_failure(SDKError(400, "Free tier is not available; billing private-secret"))
        self.assertIn("eligibility", failure.message)
        self.assertNotIn("private-secret", failure.message)

    def test_unknown_errors_do_not_leak_details_or_claim_google_rejection(self):
        failure = classify_failure(ValueError("private-secret"), "prepare")
        self.assertIsNone(failure.status)
        self.assertIn("portfolio preparation", failure.message)
        self.assertIn("without a recognized", failure.message)
        self.assertNotIn("private-secret", failure.message)

    def test_httpx_style_response_status_is_supported(self):
        error = Exception("private-secret")
        error.response = type("Response", (), {"status_code": 403})()
        self.assertEqual(classify_failure(error).status, 403)

    def test_timeout_is_distinguished_without_http_status(self):
        self.assertIn("timed out", classify_failure(TimeoutError("private-secret")).message)

    def test_cyclic_exception_chains_terminate(self):
        error = SDKError(404)
        error.__cause__ = error
        self.assertEqual(classify_failure(error).status, 404)


if __name__ == "__main__":
    unittest.main()
