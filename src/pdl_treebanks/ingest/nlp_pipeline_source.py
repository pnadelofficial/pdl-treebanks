"""First treebank source: Stanza, via nlp_pipeline's HTTP API.

Talks to nlp_pipeline over HTTP rather than importing it as a library,
mirroring MinimumViablePerseus's own src/tools/run_tokenizer.py, which
calls the same service the same way -- this keeps pdl-treebanks free of
nlp_pipeline's heavy stanza/torch dependency, and means the nlp server can
be scaled/deployed independently of the treebank build.
"""

from __future__ import annotations

import httpx

from pdl_treebanks.conllu import ConlluSentence, sentences_from_nlp_pipeline_chunk

SOURCE_NAME = "stanza:perseus"


def analyze_chunk(
    client: httpx.Client, nlp_url: str, chunk_urn: str, text: str, lang: str | None = None
) -> list[ConlluSentence]:
    """Tokenize then analyze `text` via nlp_pipeline, returning it already
    converted to pdl-treebanks' canonical ConlluSentence shape.

    Two round trips (tokenize, then analyze) rather than one, matching
    nlp_pipeline's own tokenize/analyze split -- lets a caller in principle
    reuse a tokenization done elsewhere, though pdl-treebanks doesn't yet.
    `lang`, when known ahead of time (e.g. from corpus metadata), is passed
    through explicitly so nlp_pipeline's analyze() step doesn't have to
    guess it via langid on a possibly short/ambiguous chunk.
    """
    tokenize_response = client.post(
        f"{nlp_url}/tokenize",
        json={"content": text, "lang": lang, "extra": {"urn": chunk_urn}},
    )
    tokenize_response.raise_for_status()
    tokenized_chunk = tokenize_response.json()

    analyze_response = client.post(f"{nlp_url}/analyze", json=tokenized_chunk)
    analyze_response.raise_for_status()
    analyzed_chunk = analyze_response.json()

    return sentences_from_nlp_pipeline_chunk(analyzed_chunk, chunk_urn)
