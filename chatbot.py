"""Session-owned chat orchestration, with provider-independent core logic."""

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Callable, Iterable

from settings import Settings

DATA_DIR = Path(__file__).resolve().parent / "data"
UNKNOWN = "I'm not sure based on the portfolio documents I have."


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    section: str
    start_line: int
    chunk_id: str = ""


@dataclass(frozen=True)
class Source:
    id: str
    document: Document
    similarity: float | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[Source, ...] = ()


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Split Markdown-style text into sections, preserving file and line metadata."""
    documents = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        section, lines, start = "Overview", [], 1
        headings = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"^#{1,6}\s+", line):
                text = "\n".join(lines).strip()
                if text and text != "---":
                    documents.append(Document(text, path.name, section, start))
                level = len(line) - len(line.lstrip("#"))
                title = re.sub(r"^#{1,6}\s+", "", line).replace("**", "").strip()
                headings = [(depth, heading) for depth, heading in headings if depth < level]
                headings.append((level, title))
                section = " / ".join(heading for _, heading in headings)
                lines, start = [line], number
            else:
                lines.append(line)
        text = "\n".join(lines).strip()
        if text and text != "---":
            documents.append(Document(text, path.name, section, start))
    if not documents:
        raise ValueError("No non-empty .txt portfolio documents found in data/.")
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150, add_start_index=True,
    )
    chunks = []
    for document in documents:
        for part in splitter.create_documents([document.text]):
            offset = max(0, part.metadata["start_index"])
            line = document.start_line + document.text[:offset].count("\n")
            fingerprint = f"{document.source}:{line}:{part.page_content}"
            chunks.append(Document(
                part.page_content, document.source, document.section, line,
                hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
            ))
    if not chunks:
        raise ValueError("No document chunks found.")
    return chunks


class RateLimitError(ValueError):
    pass


class RequestLimiter:
    """Thread-safe rolling budget. Contains timestamps only, never chat contents."""

    def __init__(self, limit: int, clock: Callable = time.monotonic):
        self.limit, self.clock = limit, clock
        self.timestamps = deque()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = self.clock()
            while self.timestamps and self.timestamps[0] <= now - 60:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.limit:
                raise RateLimitError("Too many requests. Please wait a minute and try again.")
            self.timestamps.append(now)


def validate_question(question: str, settings: Settings) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    if len(question) > settings.max_question_chars:
        raise ValueError(f"Keep questions under {settings.max_question_chars} characters.")
    return question


def cited_sources(text: str, sources: tuple[Source, ...]) -> tuple[Source, ...]:
    ids = set(re.findall(r"\[(S\d+)\]", text))
    return tuple(source for source in sources if source.id in ids)


class ChatService:
    """One instance per session; successful turns commit only after streaming ends."""

    def __init__(self, settings: Settings, retrieve, answer_chain, rewrite_chain,
                 process_limiter: RequestLimiter | None = None):
        self.settings = settings
        self.retrieve = retrieve
        self.answer_chain, self.rewrite_chain = answer_chain, rewrite_chain
        self.process_limiter = process_limiter
        self.session_limiter = RequestLimiter(settings.session_requests_per_minute)
        self.history: list[tuple[str, str]] = []
        self.last_answer: Answer | None = None

    def reset(self):
        self.history.clear()
        self.last_answer = None
        # Keep budgets across New Chat.

    def stream(self, question: str):
        question = validate_question(question, self.settings)
        self.last_answer = None
        self.session_limiter.acquire()
        if self.process_limiter is not None:
            self.process_limiter.acquire()
        history = list(self.history)
        query = question
        if history and self.settings.retrieval_mode == "rag":
            query = self.rewrite_chain.invoke({"history": history, "question": question}).strip()
            query = query[:self.settings.max_question_chars] or question
        candidates = self.retrieve(query)
        candidates = [
            (doc, score) for doc, score in candidates
            if score is None or (math.isfinite(score) and score >= self.settings.min_similarity)
        ]
        sources = tuple(Source(f"S{i}", doc, score)
                        for i, (doc, score) in enumerate(candidates, 1))
        pieces = []
        if not sources:
            pieces.append(UNKNOWN)
            yield UNKNOWN
        else:
            context = "\n\n".join(
                f"[{s.id}] {s.document.source} | {s.document.section}\n{s.document.text}"
                for s in sources
            )
            for piece in self.answer_chain.stream({
                "history": history, "question": question, "context": context,
            }):
                if piece:
                    pieces.append(piece)
                    yield piece
        text = "".join(pieces).strip()
        used = cited_sources(text, sources)
        # Empty/blocked responses and uncited claims never enter conversation memory.
        if not text:
            raise RuntimeError("The model returned no answer. Please try again.")
        if sources and text != UNKNOWN and not used:
            text = UNKNOWN
        allowed = {source.id for source in sources}
        if set(re.findall(r"\[(S\d+)\]", text)) - allowed:
            text, used = UNKNOWN, ()
        self.last_answer = Answer(text, used)
        self.history.extend([("human", question), ("ai", text)])
        self.history = self.history[-2 * self.settings.history_turns:]


def main():
    from dotenv import load_dotenv
    from providers import create_service, create_vectorstore
    from settings import load_settings

    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        settings = load_settings()
        chunks = split_documents(load_documents())
        index = create_vectorstore(chunks, settings) if settings.retrieval_mode == "rag" else None
        service = create_service(chunks, settings, index)
    except Exception:
        print("Could not initialize. Check configuration, portfolio files, and API access.")
        return 1
    print("Vamshi's portfolio chatbot. Type 'exit' to quit or 'new' to clear the chat.")
    while True:
        try:
            question = input("\nYou: ")
            if question.strip().lower() == "exit":
                return 0
            if question.strip().lower() == "new":
                service.reset()
                print("New chat started.")
                continue
            # Buffer terminal output so only the finalized, citation-checked answer displays.
            list(service.stream(question))
            print(f"Bot: {service.last_answer.text}")
            for source in service.last_answer.sources:
                print(f"[{source.id}] {source.document.source}: {source.document.section}")
        except (EOFError, KeyboardInterrupt):
            return 0
        except ValueError as exc:
            print(str(exc))
        except Exception:
            print("Request failed. Check API access or try again shortly.")


if __name__ == "__main__":
    raise SystemExit(main())
