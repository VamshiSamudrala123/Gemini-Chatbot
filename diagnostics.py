"""Public error diagnostics expose only approved categories, stages, and HTTP codes."""
from contextlib import contextmanager
from dataclasses import dataclass


STAGES = {
    "prepare": "portfolio preparation",
    "embedding": "portfolio embeddings",
    "setup": "model setup",
    "retrieval": "question retrieval",
    "rewrite": "follow-up rewriting",
    "generation": "answer generation",
}


class ProviderFailure(RuntimeError):
    def __init__(self, stage):
        self.stage = stage if stage in STAGES else "generation"
        super().__init__("The provider operation failed.")


@contextmanager
def provider_operation(stage):
    try:
        yield
    except Exception as exc:
        raise ProviderFailure(stage) from exc


@dataclass(frozen=True)
class Failure:
    stage: str
    status: int | None
    message: str


def classify_failure(error: Exception, stage: str = "generation") -> Failure:
    """Follow wrapped SDK causes; never return exception text, URLs, or request data."""
    status = None
    invalid_key = billing_issue = timed_out = False
    pending, visited = [error], set()
    while pending and len(visited) < 12:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, ProviderFailure):
            stage = current.stage
        for name in ("code", "status_code"):
            value = getattr(current, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and 400 <= value <= 599:
                status = status or value
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int) and 400 <= response_status <= 599:
            status = status or response_status
        # Inspect known error markers only for categorization. Never display raw text.
        try:
            text = str(current)[:8000].upper()
        except Exception:
            text = ""
        invalid_key |= "API_KEY_INVALID" in text or "API KEY NOT VALID" in text
        billing_issue |= "BILLING" in text or "FREE TIER IS NOT AVAILABLE" in text
        timed_out |= isinstance(current, TimeoutError) or type(current).__name__ in {
            "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
        }
        for cause in (current.__cause__, current.__context__):
            if cause is not None:
                pending.append(cause)

    stage = stage if stage in STAGES else "generation"
    if status == 429:
        detail = (
            "Google's quota or rate limit was reached. Check this model's limits in Google AI Studio. "
            "Retry after the applicable limit resets; a zero quota needs project/model eligibility "
            "to be resolved. Enabling payment is not automatically required."
        )
    elif status in {400, 401} and invalid_key:
        detail = "Google rejected the API key. Check or replace GOOGLE_API_KEY in Streamlit Secrets."
    elif status == 401:
        detail = "Google could not authenticate the request. Check the API key and its project."
    elif status == 403:
        detail = (
            "Google denied access. Check API-key restrictions, model access, "
            "and regional/project eligibility in Google AI Studio."
        )
    elif status == 402 or (status == 400 and billing_issue):
        detail = (
            "Google reported a billing or free-tier eligibility issue. Check your project's tier "
            "and eligibility in Google AI Studio before deciding whether to enable billing."
        )
    elif status == 404:
        detail = (
            "Google could not find the requested model or resource. "
            "Check GEMINI_CHAT_MODEL and GEMINI_EMBEDDING_MODEL against models available to your project."
        )
    elif status == 400:
        detail = "Google rejected the request parameters. Check model settings and API-key restrictions."
    elif status in {408, 504} or timed_out:
        detail = "The request timed out. Wait briefly and try again."
    elif status is not None and status >= 500:
        detail = "Google's service could not complete the request. Wait briefly and try again."
    else:
        detail = (
            "The request failed without a recognized Google HTTP status. "
            "Check model settings, portfolio files, and deployment dependencies."
        )
    label = STAGES[stage]
    code = f" (HTTP {status})" if status is not None else ""
    return Failure(stage, status, f"Failed during {label}{code}. {detail}")
