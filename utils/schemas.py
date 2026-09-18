"""Schémas Pydantic utilisés pour valider les données du pipeline RAG."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
import math

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

    score: float = Field(
        ge=-100.001,
        le=100.001,
        allow_inf_nan=False,
    )
    raw_score: float = Field(
        ge=-1.00001,
        le=1.00001,
        allow_inf_nan=False,
    )
    text: str = Field(min_length=1)
    metadata: dict[str, Any]


class EmbeddedChunk(BaseModel):
    """Chunk associé à son vecteur numérique."""

    chunk_id: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(
        cls,
        value: list[float],
    ) -> list[float]:
        """Refuse les valeurs NaN et infinies."""

        if not all(
            math.isfinite(number)
            for number in value
        ):
            raise ValueError(
                "L'embedding contient une valeur invalide."
            )

        return value


class RAGRequest(BaseModel):
    """Question envoyée au pipeline RAG."""

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Refuse une question vide ou composée d'espaces."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        return cleaned_value


class RAGResponse(BaseModel):
    """Réponse renvoyée par le pipeline RAG."""

    answer: str = Field(min_length=1)
    retrieved_contexts: list[str]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Refuse une réponse vide ou composée d'espaces."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "La réponse ne peut pas être vide."
            )

        return cleaned_value