import json
from pathlib import Path

from pdl_treebanks.aggregate import aggregate_chunk_term_counts
from pdl_treebanks.conllu import sentences_from_nlp_pipeline_chunk
from pdl_treebanks.db import TreebankDB

FIXTURE = Path(__file__).parent / "fixtures" / "iliad_two_sentences.json"
SOURCE = "stanza:perseus"


def _fixture_sentences(urn):
    chunk = json.loads(FIXTURE.read_text())
    return sentences_from_nlp_pipeline_chunk(chunk, urn)


def test_aggregate_counts_terms_and_excludes_punctuation():
    db = TreebankDB(":memory:")
    db.create_schema()

    urn = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"
    db.write_chunk(urn, SOURCE, _fixture_sentences(urn))
    db.commit()

    aggregate_chunk_term_counts(db, SOURCE)
    db.commit()

    counts = db.term_counts_for_chunk(urn, SOURCE)
    lemmas = {c["lemma"] for c in counts}

    assert "." not in lemmas  # punctuation excluded
    assert "μῆνις" in lemmas
    assert "ἀείδω" in lemmas
    assert all(c["count"] == 1 for c in counts)  # every non-punct lemma appears once here


def test_aggregate_counts_repeated_terms_within_a_chunk():
    db = TreebankDB(":memory:")
    db.create_schema()

    urn = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"
    sentences = _fixture_sentences(urn)
    db.write_chunk(urn, SOURCE, sentences + sentences)  # force every lemma to repeat
    db.commit()

    aggregate_chunk_term_counts(db, SOURCE)
    db.commit()

    counts = {c["lemma"]: c["count"] for c in db.term_counts_for_chunk(urn, SOURCE)}
    assert counts["μῆνις"] == 2
    assert counts["ἀείδω"] == 2


def test_same_lemma_feats_pair_shares_one_term_across_chunks():
    db = TreebankDB(":memory:")
    db.create_schema()

    urn_a = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"
    urn_b = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.33"
    db.write_chunk(urn_a, SOURCE, _fixture_sentences(urn_a))
    db.write_chunk(urn_b, SOURCE, _fixture_sentences(urn_b))
    db.commit()

    term_rows = db.conn.execute("SELECT lemma, feats, COUNT(*) as n FROM terms GROUP BY lemma, feats").fetchall()
    for row in term_rows:
        assert row["n"] == 1  # one terms row per distinct (lemma, feats), not one per word


def test_aggregate_is_scoped_to_the_given_source():
    db = TreebankDB(":memory:")
    db.create_schema()

    urn = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"
    db.write_chunk(urn, "stanza:perseus", _fixture_sentences(urn))
    db.write_chunk(urn, "expert:agldt", _fixture_sentences(urn))
    db.commit()

    aggregate_chunk_term_counts(db, "stanza:perseus")
    db.commit()

    assert db.term_counts_for_chunk(urn, "stanza:perseus") != []
    assert db.term_counts_for_chunk(urn, "expert:agldt") == []


def test_rerunning_aggregate_replaces_stale_counts():
    db = TreebankDB(":memory:")
    db.create_schema()

    urn = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"
    sentences = _fixture_sentences(urn)
    db.write_chunk(urn, SOURCE, sentences)
    db.commit()
    aggregate_chunk_term_counts(db, SOURCE)
    db.commit()

    db.write_chunk(urn, SOURCE, sentences[:1])  # re-ingest with fewer sentences
    db.commit()
    aggregate_chunk_term_counts(db, SOURCE)
    db.commit()

    counts = {c["lemma"] for c in db.term_counts_for_chunk(urn, SOURCE)}
    assert "Ἀχιλλεύς" not in counts  # only appeared in the now-dropped second sentence
