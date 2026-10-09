"""Streamlit portfolio chat with session-local clients and conversation history."""

from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from chatbot import RequestLimiter, UserInputError, load_documents, split_documents, validate_question
from diagnostics import classify_failure
from providers import create_service, create_vectorstore
from settings import ConfigurationError, MissingAPIKeyError, load_settings

st.set_page_config(page_title="Vamshi's Portfolio Chat", page_icon="💬", layout="centered")
load_dotenv(Path(__file__).resolve().parent / ".env")


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=2)
def load_index(chunks, settings):
    # Cache contains only the read-only index; all SDK clients and history are session-local.
    return create_vectorstore(list(chunks), settings)


@st.cache_resource(show_spinner=False)
def process_budget(limit):
    return RequestLimiter(limit)


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for source in sources:
                doc = source.document
                st.markdown(f"**[{source.id}] {doc.section}**")
                st.caption(f"{doc.source}, from line {doc.start_line}")
                st.write(doc.text)


st.title("💬 Meet Vamshi")
st.markdown("Explore my experience, projects, education, and technical skills.")
st.caption("Answers draw from my portfolio documents. Questions and relevant context are sent to Google Gemini.")

try:
    try:
        secrets = dict(st.secrets)
    except FileNotFoundError:
        secrets = {}
    settings = load_settings(secrets)
except MissingAPIKeyError as exc:
    st.info(str(exc))
    st.stop()
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
except ValueError:
    st.error("Could not read Streamlit secrets. Check the TOML syntax in the app's Settings > Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Portfolio guide")
    st.write("Ask about career experience, machine learning projects, or education.")
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        if "service" in st.session_state:
            st.session_state.service.reset()
        st.rerun()

query = None
if not st.session_state.messages:
    suggestions = [
        "What are Vamshi's technical skills?",
        "Tell me about the cancer classification capstone.",
        "What did Vamshi do at Infosys?",
    ]
    for column, suggestion in zip(st.columns(3), suggestions):
        if column.button(suggestion, use_container_width=True):
            query = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        show_sources(message.get("sources", ()))

query = st.chat_input("Ask about my portfolio…", max_chars=settings.max_question_chars) or query
if query:
    phase = "prepare"
    try:
        query = validate_question(query, settings)
        if "service" not in st.session_state or st.session_state.get("settings") != settings:
            with st.spinner("Preparing portfolio…"):
                chunks = tuple(split_documents(load_documents()))
                index = load_index(chunks, settings) if settings.retrieval_mode == "rag" else None
                phase = "setup"
                st.session_state.service = create_service(
                    chunks, settings, index, process_budget(settings.process_requests_per_minute),
                )
                st.session_state.settings = settings
        service = st.session_state.service
        phase = "generation"
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            output = st.empty()
            # SDK chunks render as they arrive. Replace partial text on failure or
            # final citation validation; never store interrupted answers as evidence.
            try:
                with output.container():
                    st.write_stream(service.stream(query))
                answer = service.last_answer
                output.markdown(answer.text)
                show_sources(answer.sources)
            except Exception:
                output.empty()
                raise
        st.session_state.messages.extend([
            {"role": "user", "text": query},
            {"role": "assistant", "text": answer.text, "sources": answer.sources},
        ])
        st.session_state.messages = st.session_state.messages[-40:]
    except UserInputError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(classify_failure(exc, phase).message)
