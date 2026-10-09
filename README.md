# Vamshi's Portfolio Chat

A Streamlit portfolio assistant using Gemini, LangChain runnables, and cosine
retrieval with FAISS. It answers questions about the supplied professional profile,
shows cited source passages, and preserves follow-up context within each session.

## Run locally

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY.
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. For terminal chat, run
`python chatbot.py`; `new` resets conversation history and `exit` quits.
Obtain a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Configuration

Environment variables and .env work locally. Streamlit secrets take precedence.
For Streamlit Community Cloud, set the repository, branch, and `app.py` as the
entry point, select Python 3.12, and put the following in app secrets:

```toml
GOOGLE_API_KEY = "your-key"
GEMINI_CHAT_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
```

Never commit .env or .streamlit/secrets.toml. Model availability and quota depend
on your Google project. These defaults were checked against Google's docs on
October 9, 2026; model IDs can change without editing application code.

| Setting | Default | Purpose |
| --- | --- | --- |
| GEMINI_CHAT_MODEL | gemini-3.8-flash | Generation and follow-up rewriting |
| GEMINI_EMBEDDING_MODEL | gemini-embedding-001 | Text retrieval embeddings |
| RETRIEVAL_MODE | rag | Set direct to send the whole small portfolio without embeddings |
| RETRIEVAL_K | 4 | Maximum retrieved chunks, from 1 to 12 |
| MIN_SIMILARITY | 0.35 | Minimum cosine similarity, from 0 to 1 |
| SESSION_REQUESTS_PER_MINUTE | 10 | Per-session submitted-question budget |
| PROCESS_REQUESTS_PER_MINUTE | 60 | Shared process submitted-question budget |

## Behavior and architecture

- The global cache contains an in-memory read-only FAISS index, keyed by portfolio
  contents and configuration, with a one-hour TTL. It contains no conversation
  history or Gemini clients. Do not load untrusted serialized FAISS/pickle files.
- Each Streamlit session owns its model clients, successful conversation turns,
  and request budget. Only six complete turns are sent as history. The UI displays
  up to twenty turns. New Chat clears both histories and keeps the budget.
- Markdown headings preserve document section, filename, line, and stable chunk
  metadata. Recursive splitting uses 1,200 characters with 150-character overlap.
  Normalized inner products are cosine similarity; higher scores are better.
- Follow-up questions are rewritten before retrieval. Below-threshold queries
  abstain without answer generation. Final answers with missing or invalid
  source IDs become an unknown response. Citation checking validates IDs, not
  whether every generated claim is entailed; prompts and evaluation are still needed.
- Response chunks render as Gemini sends them. Interrupted or empty responses
  never enter history. Streamed text is provisional until the final citation check.
  Errors use fixed messages to avoid revealing API keys or provider request details.
- RAG makes an embedding call and an answer call per accepted question; follow-ups
  also use a rewrite call. Direct mode makes one answer call and has a 30,000-character
  corpus cap. It is useful for benchmarking this small profile against RAG.
- CORS and XSRF protections remain enabled, including in Codespaces.

The rolling budgets count submitted questions, including failed attempts, and
are shared across threads in one process. They are not distributed budgets,
authentication, or a substitute for Google project quotas. Restarting the server
or opening a new session resets its corresponding budget. Configure provider quotas
and a shared gateway limit if you deploy multiple processes.

Questions, recent chat context, and relevant portfolio text go to Google Gemini.
Document text also goes to Google for embeddings in RAG mode. Only public portfolio
information belongs in data/. Chat text is not persisted to disk by the app.

## Portfolio maintenance

Add UTF-8 .txt files to data/ with descriptive Markdown headings. Restart after
changing documents so existing sessions use the new corpus. The assistant cannot
infer current employment from historical dates.

The early-risk detection claim originally said "22% accuracy" with no evaluation
details. It is now explicitly marked unverified. Confirm the intended metric,
baseline, dataset, and evaluation before publishing a numeric result.

## Checks and dependency updates

```bash
pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
pip check
pip-compile --strip-extras --output-file=requirements.txt requirements.in
```

The offline unit suite needs only the Python standard library. Integration tests
use real FAISS, LangChain runnables, SDK construction, and Streamlit AppTest with
fake providers; they make no live API requests. GitHub Actions runs these checks.
Update exact direct pins in requirements.in, regenerate the complete lock, and
rerun checks before upgrading.

## Retrieval and answer evaluation

The threshold is a starting point, not a calibrated guarantee. With your API key,
run this opt-in check, which calls Google's embeddings API and may consume quota:

```bash
python evaluate.py --k 4 --threshold 0.35
```

It prints section recall for seven factual questions plus retrieval results for
unrelated, unavailable, and unverified-metric questions. Compare k=3/4/6 and
threshold=0.3/0.35/0.45; choose using factual recall and unrelated-question rejection.
Then test these cases in both rag and direct modes, assessing factual accuracy,
source support, latency, and API usage. Test "Tell me about the Tesla project"
followed by "Which models did he use for that?" in one session; start another
session and confirm it cannot refer to the first session's conversation.

Live Gemini generation, real embedding recall, and model entitlement require a
configured key and are not covered by the credential-free test suite.

## References

- [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models)
- [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [LangChain Google integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [Streamlit resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
