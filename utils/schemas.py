"""Schémas Pydantic utilisés pour valider les données du pipeline RAG."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Métadonnées associées à un document extrait."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    category: str = Field(min_length=1)
    full_path: str = Field(min_length=1)
    sheet: str | None = None


class SourceDocument(BaseModel):
    """Document extrait avant son découpage en chunks."""

    page_content: str = Field(min_length=1)
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    """Chunk créé à partir d'un document."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any]


class SearchResult(BaseModel):
    """Résultat retourné par la recherche FAISS."""

    score: float
    raw_score: float
    text: str = Field(min_length=1)
    metadata: dict[str, Any]


class EmbeddedChunk(BaseModel):
    """Chunk associé à son vecteur numérique."""

    chunk_id: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)


class RAGRequest(BaseModel):
    """Question envoyée au pipeline RAG."""

    question: str = Field(min_length=1)


class RAGResponse(BaseModel):
    """Réponse renvoyée par le pipeline RAG."""

    answer: str = Field(min_length=1)
    retrieved_contexts: list[str]