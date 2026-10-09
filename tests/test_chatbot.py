"""Offline behavior checks using fake providers, never real API keys."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chatbot import ChatService, Document, RateLimitError, RequestLimiter, UNKNOWN, load_documents
from settings import ConfigurationError, MissingAPIKeyError, Settings, load_settings

DOC = Document("Vamshi studied Data Science at Pace University.", "data.txt", "Education", 10)


class FakeAnswer:
    def __init__(self, pieces=("Pace University [S1].",), fail=False):
        self.pieces, self.fail, self.calls = pieces, fail, []

    def stream(self, inputs):
        self.calls.append(inputs)
        yield from self.pieces
        if self.fail:
            raise RuntimeError("sensitive provider details")


class FakeRewrite:
    def __init__(self):
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return "Vamshi education at Pace University"


def make_service(score=0.9, answer=None, **kwargs):
    return ChatService(
        Settings("fake-key", **kwargs), lambda query: [(DOC, score)],
        answer or FakeAnswer(), FakeRewrite(),
    )


class ChatBehaviorTests(unittest.TestCase):
    def ask(self, service, question="Where did Vamshi study?"):
        list(service.stream(question))
        return service.last_answer

    def test_factual_answer_has_traceable_source(self):
        service = make_service()
        answer = self.ask(service)
        self.assertEqual(answer.text, "Pace University [S1].")
        self.assertEqual(answer.sources[0].document.start_line, 10)
        self.assertIn(DOC.text, service.answer_chain.calls[0]["context"])

    def test_unrelated_low_score_question_abstains_without_generation(self):
        service = make_service(score=0.05)
        answer = self.ask(service, "How do I bake sourdough?")
        self.assertEqual(answer.text, UNKNOWN)
        self.assertEqual(answer.sources, ())
        self.assertEqual(service.answer_chain.calls, [])

    def test_nonfinite_retrieval_scores_are_rejected(self):
        for score in (float("nan"), float("inf")):
            self.assertEqual(self.ask(make_service(score)).text, UNKNOWN)

    def test_follow_up_is_rewritten_using_only_own_session(self):
        service = make_service()
        self.ask(service)
        queries = []
        service.retrieve = lambda query: queries.append(query) or [(DOC, 0.9)]
        self.ask(service, "What did he study there?")
        self.assertEqual(queries, ["Vamshi education at Pace University"])
        self.assertEqual(service.rewrite_chain.calls[0]["history"][0][1], "Where did Vamshi study?")

    def test_simultaneous_sessions_do_not_share_history(self):
        first, second = make_service(), make_service()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda item: self.ask(*item),
                          [(first, "Alice's private question"), (second, "Bob's private question")]))
        self.assertNotIn("Alice", str(second.history))
        self.assertNotIn("Bob", str(first.history))
        self.assertEqual(first.answer_chain.calls[0]["history"], [])
        self.assertEqual(second.answer_chain.calls[0]["history"], [])

    def test_new_chat_clears_history_but_preserves_budget(self):
        service = make_service(session_requests_per_minute=1)
        self.ask(service)
        service.reset()
        self.assertEqual(service.history, [])
        self.assertIsNone(service.last_answer)
        with self.assertRaises(RateLimitError):
            self.ask(service)

    def test_failed_stream_does_not_commit_partial_turn(self):
        service = make_service(answer=FakeAnswer(("Partial [S1]",), fail=True))
        with self.assertRaises(RuntimeError):
            self.ask(service)
        self.assertEqual(service.history, [])
        self.assertIsNone(service.last_answer)

    def test_stream_delivers_chunks_before_commit(self):
        service = make_service(answer=FakeAnswer(("Pace ", "University [S1].")))
        stream = service.stream("Where did he study?")
        self.assertEqual(next(stream), "Pace ")
        self.assertEqual(service.history, [])
        self.assertEqual(list(stream), ["University [S1]."])
        self.assertEqual(service.last_answer.text, "Pace University [S1].")

    def test_uncited_or_invalid_citations_become_unknown(self):
        for text in ("Unsupported claim", "Pace [S99].", "Pace [S1] and fake [S9]."):
            answer = self.ask(make_service(answer=FakeAnswer((text,))))
            self.assertEqual(answer.text, UNKNOWN)
            self.assertEqual(answer.sources, ())

    def test_empty_response_is_not_committed(self):
        service = make_service(answer=FakeAnswer(()))
        with self.assertRaises(RuntimeError):
            self.ask(service)
        self.assertEqual(service.history, [])

    def test_history_is_bounded_by_complete_turns(self):
        service = make_service(history_turns=2)
        for question in ("one", "two", "three"):
            self.ask(service, question)
        self.assertEqual(len(service.history), 4)
        self.assertEqual(service.history[0], ("human", "two"))

    def test_invalid_input_uses_no_budget_or_provider(self):
        service = make_service()
        for question in (" ", "a" * 2001):
            with self.assertRaises(ValueError):
                self.ask(service, question)
        self.assertEqual(list(service.session_limiter.timestamps), [])
        self.assertEqual(service.answer_chain.calls, [])

    def test_direct_mode_skips_rewrite_and_score_threshold(self):
        service = make_service(score=None, retrieval_mode="direct")
        self.ask(service)
        self.ask(service, "What about there?")
        self.assertEqual(service.rewrite_chain.calls, [])


class InfrastructureTests(unittest.TestCase):
    def test_rate_limit_is_atomic_across_threads(self):
        limiter = RequestLimiter(3, clock=lambda: 100)

        def attempt(_):
            try:
                limiter.acquire()
                return True
            except RateLimitError:
                return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            self.assertEqual(sum(pool.map(attempt, range(20))), 3)

    def test_budget_expires_after_one_minute(self):
        now = [100]
        limiter = RequestLimiter(1, clock=lambda: now[0])
        limiter.acquire()
        with self.assertRaises(RateLimitError):
            limiter.acquire()
        now[0] = 160
        limiter.acquire()

    def test_process_budget_is_shared_between_sessions(self):
        limiter = RequestLimiter(1)
        first, second = make_service(), make_service()
        first.process_limiter = second.process_limiter = limiter
        list(first.stream("Question"))
        with self.assertRaises(RateLimitError):
            list(second.stream("Question"))

    def test_loader_preserves_nested_headings_and_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "profile.txt").write_text(
                "## Projects\n### SAM\n#### Features\nBuilding masks", encoding="utf-8",
            )
            documents = load_documents(Path(directory))
            self.assertEqual(documents[-1].start_line, 3)
            self.assertEqual(documents[-1].section, "Projects / SAM / Features")
            self.assertIn("Building masks", documents[-1].text)

    def test_loader_reports_missing_or_empty_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_documents(Path(directory))

    def test_secrets_override_environment_without_mutation(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}, clear=True):
            settings = load_settings({"GOOGLE_API_KEY": "secret-key"})
            self.assertEqual(settings.api_key, "secret-key")
            self.assertEqual(load_settings().api_key, "env-key")
            self.assertNotIn("secret-key", repr(settings))

    def test_invalid_configuration_is_rejected(self):
        for values in ({"GOOGLE_API_KEY": ""}, {"MIN_SIMILARITY": "nan"},
                       {"RETRIEVAL_K": 0}, {"RETRIEVAL_MODE": "other"}):
            with self.assertRaises(ValueError):
                load_settings({"GOOGLE_API_KEY": "fake-key", **values})
        with self.assertRaises(ValueError):
            replace(Settings("fake-key"), history_turns=0)

    def test_invalid_optional_setting_identifies_field_without_exposing_values(self):
        for name in ("RETRIEVAL_K", "MIN_SIMILARITY", "SESSION_REQUESTS_PER_MINUTE",
                     "PROCESS_REQUESTS_PER_MINUTE"):
            with self.subTest(name=name):
                with self.assertRaises(ConfigurationError) as caught:
                    load_settings({"GOOGLE_API_KEY": "private-key", name: "private-value"})
                self.assertIn(name, str(caught.exception))
                self.assertNotIn("private-key", str(caught.exception))
                self.assertNotIn("private-value", str(caught.exception))
                self.assertNotIsInstance(caught.exception, MissingAPIKeyError)

    def test_nested_key_has_actionable_diagnostic(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError) as caught:
                load_settings({"google": {"GOOGLE_API_KEY": "private-key"}})
        self.assertIn("above every [section]", str(caught.exception))
        self.assertNotIn("private-key", str(caught.exception))

    def test_blank_secret_reports_that_it_overrides_environment(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}, clear=True):
            with self.assertRaises(MissingAPIKeyError) as caught:
                load_settings({"GOOGLE_API_KEY": ""})
        self.assertIn("overrides", str(caught.exception))

    def test_top_level_secret_alone_loads_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings({"GOOGLE_API_KEY": "  private-key  "})
        self.assertEqual(settings.api_key, "private-key")
        self.assertEqual(settings.retrieval_k, 4)
        self.assertEqual(settings.retrieval_mode, "rag")

    def test_boolean_and_fractional_integer_settings_are_rejected(self):
        for value in (True, 3.5):
            with self.assertRaises(ConfigurationError):
                load_settings({"GOOGLE_API_KEY": "fake-key", "RETRIEVAL_K": value})


if __name__ == "__main__":
    unittest.main()
