"""External SDKs stay at the boundary; the shared FAISS index contains no memory."""

from dataclasses import dataclass
from typing import Any

from chatbot import ChatService, Document
from prompt import build_runnables
from settings import Settings


def create_embeddings(settings: Settings):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.api_key,
    )


@dataclass(frozen=True)
class VectorIndex:
    index: Any
    documents: tuple[Document, ...]

    def search(self, vector, k: int):
        import faiss
        import numpy as np

        values = np.asarray([vector], dtype="float32")
        if not np.isfinite(values).all() or np.linalg.norm(values) == 0:
            raise ValueError("Invalid query embedding.")
        faiss.normalize_L2(values)
        scores, positions = self.index.search(values, min(k, len(self.documents)))
        return [(self.documents[int(i)], float(score))
                for i, score in zip(positions[0], scores[0]) if i >= 0]


def create_vectorstore(chunks: list[Document], settings: Settings, embeddings=None):
    import faiss
    import numpy as np

    if not chunks:
        raise ValueError("No document chunks found.")
    embeddings = embeddings or create_embeddings(settings)
    values = np.asarray(
        embeddings.embed_documents([f"{doc.section}\n{doc.text}" for doc in chunks]), dtype="float32",
    )
    if (values.ndim != 2 or values.shape[0] != len(chunks)
            or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any()):
        raise ValueError("Invalid document embeddings.")
    faiss.normalize_L2(values)
    index = faiss.IndexFlatIP(values.shape[1])
    index.add(values)
    return VectorIndex(index, tuple(chunks))


def create_service(chunks, settings, index=None, process_limiter=None):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model, google_api_key=settings.api_key,
        temperature=0.3, max_output_tokens=1024, timeout=30, max_retries=2,
    )
    answer, rewrite = build_runnables(llm)
    if settings.retrieval_mode == "direct":
        if sum(len(doc.text) for doc in chunks) > settings.max_direct_context_chars:
            raise ValueError("Portfolio too large for direct mode; use RETRIEVAL_MODE=rag.")
        def retrieve(query):
            return [(doc, None) for doc in chunks]
    else:
        if index is None:
            raise ValueError("RAG mode requires an index.")
        # The query client belongs to this session, not the globally cached index.
        embeddings = create_embeddings(settings)
        def retrieve(query):
            return index.search(embeddings.embed_query(query), settings.retrieval_k)
    return ChatService(settings, retrieve, answer, rewrite, process_limiter)
