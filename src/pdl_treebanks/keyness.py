"""Keyness rankings, built on chunk_term_counts (pdl_treebanks.aggregate).

Two nested comparisons, as discussed:
  - work-level:   a chunk's terms, scored against the other chunks in its work
                   (TF-IDF, documents = chunks in the work)
  - author-level: a WORK's terms (its chunks pooled into one bag), scored
                   against the author's other chunks (TF-IDF, documents =
                   every chunk by the author, not "works" -- too few of
                   those per author for IDF to mean anything)

TF-IDF itself is computed in Python, not SQL: core SQLite has no
LN/LOG function without an optional build-time extension, so doing the
log here is both more portable and easier to test than relying on it
being available.

Terms appearing exactly once at the scope being ranked are excluded from
the ranked list and reported separately by hapax_tiers() instead -- by
design (see project discussion): a term used once isn't a *pattern* within
that scope, and burying it at the top of every ranking (which raw TF-IDF
would do, since a hapax's IDF is always high) drowns out real signal.
"""

from __future__ import annotations

import math

from pdl_treebanks.db import TreebankDB

MIN_COUNT_FOR_RANKING = 2


def _chunk_id(db: TreebankDB, cts_urn: str, source: str) -> int | None:
    row = db.conn.execute(
        """
        SELECT chunks.id FROM chunks
        JOIN sources ON sources.id = chunks.source_id
        WHERE chunks.cts_urn = ? AND sources.name = ?
        """,
        (cts_urn, source),
    ).fetchone()
    return row["id"] if row is not None else None


def _work_of_chunk(db: TreebankDB, chunk_id: int) -> tuple[str, str, int]:
    row = db.conn.execute(
        "SELECT text_group, work, source_id FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    return row["text_group"], row["work"], row["source_id"]


def _term_rows(db: TreebankDB, term_ids: list[int]) -> dict[int, tuple[str, str | None]]:
    if not term_ids:
        return {}
    placeholders = ",".join("?" * len(term_ids))
    rows = db.conn.execute(
        f"SELECT id, lemma, feats FROM terms WHERE id IN ({placeholders})", term_ids
    ).fetchall()
    return {row["id"]: (row["lemma"], row["feats"]) for row in rows}


def work_level_keywords(db: TreebankDB, cts_urn: str, source: str) -> list[dict]:
    """This chunk's terms, TF-IDF-ranked against the other chunks in its
    work. Excludes terms occurring only once in this chunk (see module
    docstring) -- those show up in hapax_tiers() instead. Returns [] if the
    chunk hasn't been ingested or aggregated for `source`."""
    chunk_id = _chunk_id(db, cts_urn, source)
    if chunk_id is None:
        return []
    text_group, work, source_id = _work_of_chunk(db, chunk_id)

    work_chunk_count = db.conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE text_group = ? AND work = ? AND source_id = ?",
        (text_group, work, source_id),
    ).fetchone()["n"]

    doc_freq = {
        row["term_id"]: row["df"]
        for row in db.conn.execute(
            """
            SELECT chunk_term_counts.term_id AS term_id, COUNT(DISTINCT chunk_term_counts.chunk_id) AS df
            FROM chunk_term_counts
            JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
            WHERE chunks.text_group = ? AND chunks.work = ? AND chunks.source_id = ?
            GROUP BY chunk_term_counts.term_id
            """,
            (text_group, work, source_id),
        ).fetchall()
    }

    chunk_counts = db.conn.execute(
        "SELECT term_id, count FROM chunk_term_counts WHERE chunk_id = ?", (chunk_id,)
    ).fetchall()
    chunk_total = sum(row["count"] for row in chunk_counts)
    if chunk_total == 0:
        return []

    ranked = []
    for row in chunk_counts:
        if row["count"] < MIN_COUNT_FOR_RANKING:
            continue
        tf = row["count"] / chunk_total
        idf = math.log(work_chunk_count / doc_freq[row["term_id"]])
        ranked.append((row["term_id"], row["count"], tf * idf))

    terms = _term_rows(db, [r[0] for r in ranked])
    return [
        {"lemma": terms[term_id][0], "feats": terms[term_id][1], "count": count, "tfidf": tfidf}
        for term_id, count, tfidf in sorted(ranked, key=lambda r: r[2], reverse=True)
    ]


def author_level_keywords(db: TreebankDB, text_group: str, work: str, source: str) -> list[dict]:
    """A work's terms (its chunks pooled together), TF-IDF-ranked against
    every other chunk by the same author (not "the author's other works" --
    see module docstring for why documents are chunks, not works). Excludes
    terms occurring only once across the whole work."""
    source_id_row = db.conn.execute("SELECT id FROM sources WHERE name = ?", (source,)).fetchone()
    if source_id_row is None:
        return []
    source_id = source_id_row["id"]

    author_chunk_count = db.conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE text_group = ? AND source_id = ?",
        (text_group, source_id),
    ).fetchone()["n"]

    doc_freq = {
        row["term_id"]: row["df"]
        for row in db.conn.execute(
            """
            SELECT chunk_term_counts.term_id AS term_id, COUNT(DISTINCT chunk_term_counts.chunk_id) AS df
            FROM chunk_term_counts
            JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
            WHERE chunks.text_group = ? AND chunks.source_id = ?
            GROUP BY chunk_term_counts.term_id
            """,
            (text_group, source_id),
        ).fetchall()
    }

    work_counts = db.conn.execute(
        """
        SELECT chunk_term_counts.term_id AS term_id, SUM(chunk_term_counts.count) AS total
        FROM chunk_term_counts
        JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
        WHERE chunks.text_group = ? AND chunks.work = ? AND chunks.source_id = ?
        GROUP BY chunk_term_counts.term_id
        """,
        (text_group, work, source_id),
    ).fetchall()
    work_total = sum(row["total"] for row in work_counts)
    if work_total == 0:
        return []

    ranked = []
    for row in work_counts:
        if row["total"] < MIN_COUNT_FOR_RANKING:
            continue
        tf = row["total"] / work_total
        idf = math.log(author_chunk_count / doc_freq[row["term_id"]])
        ranked.append((row["term_id"], row["total"], tf * idf))

    terms = _term_rows(db, [r[0] for r in ranked])
    return [
        {"lemma": terms[term_id][0], "feats": terms[term_id][1], "count": count, "tfidf": tfidf}
        for term_id, count, tfidf in sorted(ranked, key=lambda r: r[2], reverse=True)
    ]


def hapax_tiers(db: TreebankDB, cts_urn: str, source: str) -> dict[str, list[dict]]:
    """Terms occurring exactly once in this chunk, tagged with the
    *tightest* scope they're unique at: 'author' (this is the author's only
    use of the term, ever) beats 'work' (used elsewhere in the work, but
    not this chunk... wait: unique to this work as a whole) beats 'passage'
    (unique here, but recurs elsewhere in the work). Each term appears in
    exactly one of the three lists."""
    chunk_id = _chunk_id(db, cts_urn, source)
    if chunk_id is None:
        return {"passage": [], "work": [], "author": []}
    text_group, work, source_id = _work_of_chunk(db, chunk_id)

    chunk_hapax_term_ids = [
        row["term_id"]
        for row in db.conn.execute(
            "SELECT term_id FROM chunk_term_counts WHERE chunk_id = ? AND count = 1", (chunk_id,)
        ).fetchall()
    ]
    if not chunk_hapax_term_ids:
        return {"passage": [], "work": [], "author": []}

    placeholders = ",".join("?" * len(chunk_hapax_term_ids))

    work_totals = {
        row["term_id"]: row["total"]
        for row in db.conn.execute(
            f"""
            SELECT chunk_term_counts.term_id AS term_id, SUM(chunk_term_counts.count) AS total
            FROM chunk_term_counts
            JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
            WHERE chunks.text_group = ? AND chunks.work = ? AND chunks.source_id = ?
              AND chunk_term_counts.term_id IN ({placeholders})
            GROUP BY chunk_term_counts.term_id
            """,
            (text_group, work, source_id, *chunk_hapax_term_ids),
        ).fetchall()
    }
    author_totals = {
        row["term_id"]: row["total"]
        for row in db.conn.execute(
            f"""
            SELECT chunk_term_counts.term_id AS term_id, SUM(chunk_term_counts.count) AS total
            FROM chunk_term_counts
            JOIN chunks ON chunks.id = chunk_term_counts.chunk_id
            WHERE chunks.text_group = ? AND chunks.source_id = ?
              AND chunk_term_counts.term_id IN ({placeholders})
            GROUP BY chunk_term_counts.term_id
            """,
            (text_group, source_id, *chunk_hapax_term_ids),
        ).fetchall()
    }

    terms = _term_rows(db, chunk_hapax_term_ids)
    tiers: dict[str, list[dict]] = {"passage": [], "work": [], "author": []}
    for term_id in chunk_hapax_term_ids:
        lemma, feats = terms[term_id]
        entry = {"lemma": lemma, "feats": feats}
        if author_totals.get(term_id) == 1:
            tiers["author"].append(entry)
        elif work_totals.get(term_id) == 1:
            tiers["work"].append(entry)
        else:
            tiers["passage"].append(entry)

    return tiers
