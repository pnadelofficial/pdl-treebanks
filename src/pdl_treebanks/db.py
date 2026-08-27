"""SQLite storage for prebuilt treebank data.

Schema shape borrows from two existing PDL projects rather than inventing a
third pattern: pdl-morph-server's "ingest offline into SQLite, serve reads
live over HTTP" split, and sanjaya's AnnotationDB convention of keeping only
load-bearing fields as real columns and pushing anything else into a
generic key/value side table (word_features) so a new field never needs a
migration -- deliberately not needed yet (single source, Stanza via
nlp_pipeline) but the schema is shaped to make adding a second source
(spaCy, expert annotation) additive rather than a rewrite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pdl_treebanks.conllu import ConlluSentence, ConlluWord
from pdl_treebanks.cts import parse_cts_urn

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    cts_urn TEXT NOT NULL,
    text_group TEXT NOT NULL,
    work TEXT NOT NULL,
    version TEXT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    UNIQUE(cts_urn, source_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_text_group ON chunks(text_group);
CREATE INDEX IF NOT EXISTS idx_chunks_text_group_work ON chunks(text_group, work);

CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    sent_index INTEGER NOT NULL,
    sent_id TEXT NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(chunk_id, sent_index)
);

CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY,
    sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL,
    conllu_id TEXT NOT NULL,
    urn TEXT,
    form TEXT NOT NULL,
    lemma TEXT,
    upos TEXT,
    xpos TEXT,
    feats TEXT,
    head INTEGER,
    deprel TEXT,
    deps TEXT,
    misc TEXT,
    term_id INTEGER REFERENCES terms(id),
    UNIQUE(sentence_id, word_index)
);

CREATE TABLE IF NOT EXISTS word_features (
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_word_features_key_value ON word_features(key, value);
CREATE INDEX IF NOT EXISTS idx_words_urn ON words(urn);
CREATE INDEX IF NOT EXISTS idx_words_lemma ON words(lemma);
CREATE INDEX IF NOT EXISTS idx_words_term ON words(term_id);

-- Vocabulary for keyness/frequency analysis: one row per distinct
-- (lemma, feats) pairing -- a morphologically-disambiguated lemma, e.g.
-- "used as an accusative singular" vs. "used as a genitive plural" count
-- as different terms even with the same lemma. feats is nullable (some
-- words, e.g. particles/adpositions, carry no morphological features);
-- looked up with `feats IS ?` rather than relying on the UNIQUE
-- constraint alone, since SQL NULLs never compare equal to each other.
CREATE TABLE IF NOT EXISTS terms (
    id INTEGER PRIMARY KEY,
    lemma TEXT NOT NULL,
    feats TEXT,
    UNIQUE(lemma, feats)
);

-- Precomputed per-chunk term counts -- the one expensive-to-derive
-- (requires scanning every word) aggregate, populated by
-- pdl_treebanks.aggregate as a separate offline step, mirroring
-- pdl-morph-server's own ingest/aggregate split. Work-level and
-- author-level keyness (TF-IDF, hapax tiers) are computed by grouping
-- this compact table by chunks.work / chunks.text_group at query time
-- rather than precomputing further layers -- cheap once this table
-- exists, since it's already a small per-chunk summary, not the full
-- words table.
CREATE TABLE IF NOT EXISTS chunk_term_counts (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(id),
    count INTEGER NOT NULL,
    UNIQUE(chunk_id, term_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_term_counts_term ON chunk_term_counts(term_id);
"""


class TreebankDB:
    def __init__(self, path: str | Path):
        # check_same_thread=False: the FastAPI server (pdl_treebanks.server)
        # opens one TreebankDB at import time and serves every request
        # (each dispatched to a worker thread by Starlette) through it.
        # Safe here because the server only ever reads -- writes happen
        # offline, from a separate TreebankDB instance in the ingest CLI.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def create_schema(self) -> None:
        self.conn.executescript(SCHEMA)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _source_id(self, source: str) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO sources (name) VALUES (?)", (source,)
        )
        row = self.conn.execute(
            "SELECT id FROM sources WHERE name = ?", (source,)
        ).fetchone()
        return row["id"]

    def _term_id(self, lemma: str, feats: str | None) -> int:
        # Not INSERT OR IGNORE + UNIQUE(lemma, feats): SQL NULLs never
        # compare equal, so the UNIQUE constraint alone wouldn't dedupe
        # multiple rows for the same lemma with feats IS NULL. `feats IS ?`
        # is SQLite's NULL-safe equality, so this works whether feats is
        # a real string or None.
        row = self.conn.execute(
            "SELECT id FROM terms WHERE lemma = ? AND feats IS ?", (lemma, feats)
        ).fetchone()
        if row is not None:
            return row["id"]
        cursor = self.conn.execute(
            "INSERT INTO terms (lemma, feats) VALUES (?, ?)", (lemma, feats)
        )
        return cursor.lastrowid

    def write_chunk(
        self, cts_urn: str, source: str, sentences: list[ConlluSentence]
    ) -> None:
        """Replace any existing (cts_urn, source) chunk with `sentences`.
        Not auto-committed -- call commit() once per batch, matching
        sanjaya's AnnotationDB convention of one commit per unit of work
        rather than per row."""
        urn_parts = parse_cts_urn(cts_urn)
        source_id = self._source_id(source)

        existing = self.conn.execute(
            "SELECT id FROM chunks WHERE cts_urn = ? AND source_id = ?",
            (cts_urn, source_id),
        ).fetchone()
        if existing is not None:
            self.conn.execute("DELETE FROM chunks WHERE id = ?", (existing["id"],))

        cursor = self.conn.execute(
            "INSERT INTO chunks (cts_urn, text_group, work, version, source_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (cts_urn, urn_parts.text_group, urn_parts.work, urn_parts.version, source_id),
        )
        chunk_id = cursor.lastrowid

        for sent_index, sentence in enumerate(sentences):
            self._write_sentence(chunk_id, sent_index, sentence)

    def _write_sentence(
        self, chunk_id: int, sent_index: int, sentence: ConlluSentence
    ) -> None:
        cursor = self.conn.execute(
            "INSERT INTO sentences (chunk_id, sent_index, sent_id, text) "
            "VALUES (?, ?, ?, ?)",
            (chunk_id, sent_index, sentence.sent_id, sentence.text),
        )
        sentence_id = cursor.lastrowid

        for word_index, word in enumerate(sentence.words):
            self._write_word(sentence_id, word_index, word)

    def _write_word(self, sentence_id: int, word_index: int, word: ConlluWord) -> None:
        # term_id is None for range lines (MWT) and any word with no lemma
        # (e.g. a malformed/unanalyzed token) -- there's no vocabulary entry
        # to attach in either case.
        term_id = self._term_id(word.lemma, word.feats) if word.lemma else None

        self.conn.execute(
            """
            INSERT INTO words (
                sentence_id, word_index, conllu_id, urn, form, lemma,
                upos, xpos, feats, head, deprel, deps, misc, term_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sentence_id,
                word_index,
                word.id,
                word.urn,
                word.form,
                word.lemma,
                word.upos,
                word.xpos,
                word.feats,
                word.head,
                word.deprel,
                word.deps,
                word.misc_str(),
                term_id,
            ),
        )

    def read_chunk(self, cts_urn: str, source: str) -> list[ConlluSentence] | None:
        chunk = self.conn.execute(
            """
            SELECT chunks.id FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            WHERE chunks.cts_urn = ? AND sources.name = ?
            """,
            (cts_urn, source),
        ).fetchone()
        if chunk is None:
            return None

        sentence_rows = self.conn.execute(
            "SELECT * FROM sentences WHERE chunk_id = ? ORDER BY sent_index",
            (chunk["id"],),
        ).fetchall()

        sentences = []
        for sent_row in sentence_rows:
            word_rows = self.conn.execute(
                "SELECT * FROM words WHERE sentence_id = ? ORDER BY word_index",
                (sent_row["id"],),
            ).fetchall()
            words = [
                ConlluWord(
                    id=w["conllu_id"],
                    form=w["form"],
                    lemma=w["lemma"],
                    upos=w["upos"],
                    xpos=w["xpos"],
                    feats=w["feats"],
                    head=w["head"],
                    deprel=w["deprel"],
                    deps=w["deps"],
                    misc=_parse_misc(w["misc"]),
                    urn=w["urn"],
                )
                for w in word_rows
            ]
            sentences.append(
                ConlluSentence(sent_id=sent_row["sent_id"], text=sent_row["text"], words=words)
            )

        return sentences

    def chunks_for_work(self, text_group: str, work: str, source: str) -> list[str]:
        """CTS URNs of every ingested chunk for (text_group, work, source),
        in ingestion order. E.g. chunks_for_work("tlg0012", "tlg001",
        "stanza:perseus") -> every Iliad chunk Stanza has analyzed."""
        rows = self.conn.execute(
            """
            SELECT chunks.cts_urn FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            WHERE chunks.text_group = ? AND chunks.work = ? AND sources.name = ?
            ORDER BY chunks.id
            """,
            (text_group, work, source),
        ).fetchall()
        return [row["cts_urn"] for row in rows]

    def chunks_for_text_group(self, text_group: str, source: str) -> list[str]:
        """CTS URNs of every ingested chunk for a text_group (author) across
        all of its works, in ingestion order."""
        rows = self.conn.execute(
            """
            SELECT chunks.cts_urn FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            WHERE chunks.text_group = ? AND sources.name = ?
            ORDER BY chunks.id
            """,
            (text_group, source),
        ).fetchall()
        return [row["cts_urn"] for row in rows]

    def term_counts_for_chunk(self, cts_urn: str, source: str) -> list[dict]:
        """(lemma, feats, count) for every counted term in a chunk, highest
        count first. Requires pdl_treebanks.aggregate.aggregate_chunk_term_counts
        to have been run for `source` -- returns [] if it hasn't (or if the
        chunk is all punctuation/lemma-less words)."""
        rows = self.conn.execute(
            """
            SELECT terms.lemma, terms.feats, chunk_term_counts.count
            FROM chunk_term_counts
            JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
            JOIN sources ON sources.id = chunks.source_id
            JOIN terms ON terms.id = chunk_term_counts.term_id
            WHERE chunks.cts_urn = ? AND sources.name = ?
            ORDER BY chunk_term_counts.count DESC, terms.lemma
            """,
            (cts_urn, source),
        ).fetchall()
        return [
            {"lemma": row["lemma"], "feats": row["feats"], "count": row["count"]}
            for row in rows
        ]


def _parse_misc(misc_str: str | None) -> dict[str, str]:
    if not misc_str or misc_str == "_":
        return {}
    return dict(pair.split("=", 1) for pair in misc_str.split("|"))
