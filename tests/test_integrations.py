"""Real dependency smoke tests with deterministic offline embeddings/models."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from chatbot import Document, load_documents, split_documents
from providers import create_service, create_vectorstore
from settings import Settings


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[2.0, 0.0], [0.0, 3.0]]

    def embed_query(self, text):
        return [2.0, 0.0]


class IntegrationTests(unittest.TestCase):
    def test_faiss_returns_cosine_similarity_and_correct_ranking(self):
        docs = [Document("Education", "data.txt", "Education", 1),
                Document("Projects", "data.txt", "Projects", 2)]
        index = create_vectorstore(docs, Settings("fake-key"), FakeEmbeddings())
        results = index.search([5.0, 0.0], 4)
        self.assertEqual(results[0][0], docs[0])
        self.assertAlmostEqual(results[0][1], 1.0, places=5)
        self.assertAlmostEqual(results[1][1], 0.0, places=5)
        with self.assertRaises(ValueError):
            index.search([float("nan"), 0], 1)

    def test_splitter_limits_chunks_and_preserves_source_metadata(self):
        docs = [Document("long paragraph " * 600, "profile.txt", "Projects / SAM", 20)]
        chunks = split_documents(docs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 1200 for chunk in chunks))
        self.assertTrue(all(chunk.source == "profile.txt" for chunk in chunks))
        self.assertEqual(chunks[0].section, "Projects / SAM")
        self.assertEqual(chunks, split_documents(docs))

    def test_runnables_support_real_messages_and_stream_parsing(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from prompt import build_runnables
        answer, rewrite = build_runnables(FakeListChatModel(responses=["Pace [S1]", "Pace education"]))
        params = {"history": [("human", "Where?"), ("ai", "Pace [S1]")],
                  "context": "[S1] Pace", "question": "What did he study?"}
        self.assertEqual("".join(answer.stream(params)), "Pace [S1]")
        self.assertEqual(rewrite.invoke(params), "Pace education")

    def test_provider_clients_initialize_without_network(self):
        chunks = split_documents(load_documents())
        direct = create_service(chunks, Settings("fake-key", retrieval_mode="direct"))
        self.assertEqual(len(direct.retrieve("Anything")), len(chunks))
        with patch("providers.create_embeddings", return_value=FakeEmbeddings()):
            index = create_vectorstore(chunks[:2], Settings("fake-key"), FakeEmbeddings())
            rag = create_service(chunks[:2], Settings("fake-key"), index)
            self.assertAlmostEqual(rag.retrieve("Education")[0][1], 1.0, places=5)

    def test_ui_shows_setup_message_without_credentials(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

    def test_ui_reads_a_top_level_streamlit_secret(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.secrets["GOOGLE_API_KEY"] = "private-key"
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.info)
        self.assertTrue(app.chat_input)

    def test_ui_invalid_optional_setting_does_not_claim_key_is_missing(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.secrets["GOOGLE_API_KEY"] = "private-key"
            app.secrets["RETRIEVAL_K"] = "private-invalid-value"
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.info)
        self.assertIn("RETRIEVAL_K", app.error[0].value)
        self.assertNotIn("private-key", app.error[0].value)
        self.assertNotIn("private-invalid-value", app.error[0].value)

    def test_ui_nested_secret_explains_top_level_requirement(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.secrets["google"] = {"GOOGLE_API_KEY": "private-key"}
            app.run()
        self.assertFalse(app.exception)
        self.assertIn("TOML section", app.error[0].value)
        self.assertNotIn("private-key", app.error[0].value)

    def test_ui_chat_and_new_chat_use_session_state(self):
        from streamlit.testing.v1 import AppTest
        from test_chatbot import make_service
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}, clear=True):
            with patch("providers.create_vectorstore", return_value=object()), patch(
                "providers.create_service", side_effect=lambda *a, **kw: make_service(),
            ):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
                self.assertFalse(app.exception)
                app.chat_input[0].set_value("Where did Vamshi study?").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state["messages"]), 2)
                self.assertIn("Pace", app.session_state["messages"][1]["text"])
                app.sidebar.button[0].click().run()
                self.assertEqual(app.session_state["messages"], [])
                self.assertEqual(app.session_state["service"].history, [])


if __name__ == "__main__":
    unittest.main()
