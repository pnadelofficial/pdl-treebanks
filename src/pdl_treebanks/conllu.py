"""CoNLL-U data model and conversion from nlp_pipeline's TokenizedChunk JSON.

pdl-treebanks' canonical internal shape is CoNLL-U (see project decisions):
one strong unified schema, with per-source conversion into it rather than
reconciling heterogeneous sources at build/serve time. This module defines
that shape (ConlluWord/ConlluSentence) and the first conversion function,
from nlp_pipeline (Stanza) output.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

_EMPTY = "_"


def _s(value: Any) -> str:
    return _EMPTY if value in (None, "") else str(value)


def _is_punct(text: str) -> bool:
    return bool(text) and all(unicodedata.category(c)[0] in ("P", "S") for c in text)


@dataclass
class ConlluWord:
    """One CoNLL-U line: a single word, or a multiword-token range line
    (id like "2-3", form set, every other field "_")."""

    id: str
    form: str
    lemma: str | None = None
    upos: str | None = None
    xpos: str | None = None
    feats: str | None = None
    head: int | None = None
    deprel: str | None = None
    deps: str | None = None
    misc: dict[str, str] = field(default_factory=dict)
    # Per-surface-token CTS URN (e.g. "urn:cts:...@word[3]"), shared by a
    # multiword-token range line and the word lines it expands into. None
    # for punctuation, matching MinimumViablePerseus's run_tokenizer.py
    # convention of not assigning citable URNs to punctuation tokens.
    urn: str | None = None

    def is_range(self) -> bool:
        return "-" in self.id

    def misc_str(self) -> str:
        if not self.misc:
            return _EMPTY
        return "|".join(f"{k}={v}" for k, v in sorted(self.misc.items()))

    def to_conllu_fields(self) -> list[str]:
        if self.is_range():
            return [self.id, self.form, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, self.misc_str()]
        return [
            self.id,
            self.form,
            _s(self.lemma),
            _s(self.upos),
            _s(self.xpos),
            _s(self.feats),
            _s(self.head),
            _s(self.deprel),
            _s(self.deps),
            self.misc_str(),
        ]


@dataclass
class ConlluSentence:
    sent_id: str
    text: str
    words: list[ConlluWord]

    def to_text(self) -> str:
        lines = [f"# sent_id = {self.sent_id}", f"# text = {self.text}"]
        lines += ["\t".join(w.to_conllu_fields()) for w in self.words]
        return "\n".join(lines) + "\n"


def _group_into_sentences(tokens: list[dict]) -> list[list[dict]]:
    """Split nlp_pipeline's flat token list back into per-sentence groups.

    nlp_pipeline's TokenizedChunk has no explicit sentence-boundary field —
    Stanza numbers word ids 1..n *within* each sentence and restarts at 1
    for the next, but that structure is discarded when the Document is
    flattened via doc.iter_tokens() into one flat `tokens` list (see
    nlp_pipeline.pipeline._collect_tokens). This reconstructs sentence
    boundaries from that implicit signal: whenever the next word's id isn't
    the previous word's id + 1 (a restart, e.g. 4 -> 1, or in principle any
    non-consecutive jump), a new sentence has started. Reliable because
    Stanza always numbers each sentence's words/MWT-expansions consecutively
    from 1, but this is an inferred contract, not one nlp_pipeline's schema
    documents or guarantees — worth raising upstream if it ever changes.
    """
    sentences: list[list[dict]] = []
    current: list[dict] = []
    prev_last_word_id: int | None = None

    for token in tokens:
        first_word_id = token["words"][0]["id"]
        if current and first_word_id != prev_last_word_id + 1:
            sentences.append(current)
            current = []
        current.append(token)
        prev_last_word_id = token["words"][-1]["id"]

    if current:
        sentences.append(current)

    return sentences


def _token_misc(token: dict) -> dict[str, str]:
    misc = {}
    if not token.get("whitespace", True):
        misc["SpaceAfter"] = "No"
    if token.get("start_char") is not None:
        misc["start_char"] = str(token["start_char"])
    if token.get("end_char") is not None:
        misc["end_char"] = str(token["end_char"])
    return misc


def sentences_from_nlp_pipeline_chunk(
    tokenized_chunk: dict, chunk_urn: str
) -> list[ConlluSentence]:
    """Convert an nlp_pipeline TokenizedChunk (the dict shape returned by its
    /analyze endpoint or NLPPipeline.analyze().model_dump()) into
    pdl-treebanks' canonical ConlluSentence list, assigning each non-punct
    token a per-token CTS URN of `{chunk_urn}@{token['identifier']}` —
    matching the convention MinimumViablePerseus's run_tokenizer.py already
    uses for morphological tokens, so treebank tokens and morph-lookup
    tokens are addressable the same way.
    """
    sentences: list[ConlluSentence] = []

    for sent_index, sent_tokens in enumerate(_group_into_sentences(tokenized_chunk["tokens"]), start=1):
        words: list[ConlluWord] = []
        text_parts: list[str] = []

        for token in sent_tokens:
            text = token["text"].strip()
            token_urn = None if _is_punct(text) else f"{chunk_urn}@{token['identifier']}"
            token_misc = _token_misc(token)

            text_parts.append(token["text"])
            if token.get("whitespace"):
                text_parts.append(" ")

            token_words = token["words"]
            is_mwt = len(token_words) > 1

            # SpaceAfter/start_char/end_char describe the surface token, so
            # they belong on the range line for a multiword token; a
            # single-word token has no separate range line, so they go on
            # its one word line instead.
            if is_mwt:
                first_id, last_id = token_words[0]["id"], token_words[-1]["id"]
                words.append(
                    ConlluWord(
                        id=f"{first_id}-{last_id}",
                        form=text,
                        misc=token_misc,
                        urn=token_urn,
                    )
                )

            for word in token_words:
                words.append(
                    ConlluWord(
                        id=str(word["id"]),
                        form=word["text"],
                        lemma=word.get("lemma"),
                        upos=word.get("upos"),
                        xpos=word.get("xpos"),
                        feats=word.get("feats"),
                        head=word.get("head"),
                        deprel=word.get("deprel"),
                        deps=word.get("deps"),
                        misc={} if is_mwt else token_misc,
                        urn=token_urn,
                    )
                )

        sentences.append(
            ConlluSentence(
                sent_id=f"{chunk_urn}:{sent_index}",
                text="".join(text_parts).strip(),
                words=words,
            )
        )

    return sentences
