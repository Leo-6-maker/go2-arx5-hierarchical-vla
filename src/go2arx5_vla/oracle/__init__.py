"""Deterministic, zero-model oracle-corpus building blocks."""

from .stage35_corpus import (
    CASES_PER_CLASS,
    CORPUS_CLASSES,
    CORPUS_SAMPLE_COUNT,
    DURATION_S,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    build_case,
    case_ids,
    protocol_manifest,
)

__all__ = [
    "CASES_PER_CLASS",
    "CORPUS_CLASSES",
    "CORPUS_SAMPLE_COUNT",
    "DURATION_S",
    "SOURCE_FRAME",
    "TCP_SEMANTIC_ID",
    "build_case",
    "case_ids",
    "protocol_manifest",
]
