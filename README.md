# Vamshi's Portfolio Chat

A chat-based way to explore my experience in AI, data engineering, and analytics.
The assistant answers questions from my resume and links each answer to the relevant
source passages.

Built with Python, Streamlit, Gemini, LangChain, and FAISS.

## Features

- Resume-based answers with visible sources
- Follow-up questions and separate history for each visitor
- Streaming responses, suggested questions, and a New Chat button
- Clear setup and API error messages
- RAG retrieval and a direct-context mode for smaller documents

## Run locally

Requires Python 3.12 and a [Google AI Studio API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`.
For terminal chat, run `python chatbot.py`; `new` clears the conversation and `exit` quits.

## Deploy on Streamlit

Select this repository, the `main` branch, `app.py`, and Python 3.12.
Add `GOOGLE_API_KEY = "your-key"` at the top level of the app's Secrets settings.
Optional settings are listed in `.env.example`.

Streamlit secrets override local environment values. Model access and quotas depend
on your Google project. Choose model IDs available to that project.
Keep `.env` and `.streamlit/secrets.toml` out of Git.

## How it works

The app splits `data/resume.txt` into sections and overlapping chunks. In RAG mode,
Gemini embeddings and FAISS retrieve relevant passages before Gemini answers.
Follow-up questions are rewritten using the visitor's recent conversation.

Answers include source IDs and expandable passages. Missing or invalid citation IDs
produce an unknown answer; this check does not verify every claim in an answer.
Direct mode sends the whole resume and skips embeddings and follow-up rewriting.

Each visitor has separate conversation history and model clients. The shared cache
holds only the read-only document index. Updating the resume clears old conversation
context on the next app rerun; reboot after deploying new application code.

Questions and relevant resume text are sent to Google. Chat messages are kept in
session memory and are not saved to disk. Request limits apply per session and per
server process.

## Update the background

Edit `data/resume.txt` with the latest resume information, keeping descriptive headings.
The resume is the source for employment, education, skills, and reported results.
Answers should acknowledge information that the resume does not provide.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
pip check
```

Tests cover chat behavior, session isolation, source handling, configuration,
provider errors, and Streamlit flows without live API calls.
GitHub Actions runs the checks on pushes and pull requests.

To update dependencies, edit `requirements.in`, regenerate the lock, and rerun the checks:

```bash
pip-compile --strip-extras --output-file=requirements.txt requirements.in
```

To evaluate retrieval with real embeddings:

```bash
python evaluate.py --k 4 --threshold 0.35
```

This uses Google's API and may consume quota. Review factual answers, follow-ups,
unknown questions, source support, latency, and API usage before deployment.
