"""Offline aggregation: derived statistics computed from already-ingested
treebank data, kept as a separate step from ingestion (pdl_treebanks.ingest)
-- mirrors pdl-morph-server's own load/aggregate/ingest pipeline split.

Currently computes chunk_term_counts: for every chunk, how many times each
(lemma, feats) term occurs. That's the one expensive-to-derive layer (it
has to scan every word) that work-level and author-level keyness --
TF-IDF, hapax tiers -- both build on, by grouping this compact table over
chunks.work / chunks.text_group at query time rather than precomputing
further layers, since aggregating a few hundred (chunk, term) rows per
chunk is cheap once this table exists.

Usage:
    pdl-treebanks-aggregate --db treebanks.db --source stanza:perseus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdl_treebanks.db import TreebankDB


def aggregate_chunk_term_counts(db: TreebankDB, source: str) -> int:
    """(Re)compute chunk_term_counts for every chunk of `source`. Excludes
    punctuation (upos = 'PUNCT') and lemma-less words -- neither is
    meaningful vocabulary for frequency/keyness analysis. Returns the
    number of chunks processed."""
    chunk_rows = db.conn.execute(
        """
        SELECT chunks.id FROM chunks
        JOIN sources ON sources.id = chunks.source_id
        WHERE sources.name = ?
        """,
        (source,),
    ).fetchall()

    for chunk_row in chunk_rows:
        chunk_id = chunk_row["id"]
        db.conn.execute("DELETE FROM chunk_term_counts WHERE chunk_id = ?", (chunk_id,))
        db.conn.execute(
            """
            INSERT INTO chunk_term_counts (chunk_id, term_id, count)
            SELECT ?, words.term_id, COUNT(*)
            FROM words
            JOIN sentences ON sentences.id = words.sentence_id
            WHERE sentences.chunk_id = ?
              AND words.term_id IS NOT NULL
              AND words.upos IS NOT 'PUNCT'
            GROUP BY words.term_id
            """,
            (chunk_id, chunk_id),
        )

    return len(chunk_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate derived stats for a pdl-treebanks db")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source", required=True, help='e.g. "stanza:perseus"')
    args = parser.parse_args()

    db = TreebankDB(args.db)
    db.create_schema()
    count = aggregate_chunk_term_counts(db, args.source)
    db.commit()
    db.close()

    print(f"chunk_term_counts: {count} chunks aggregated for source {args.source!r}.")


if __name__ == "__main__":
    main()
