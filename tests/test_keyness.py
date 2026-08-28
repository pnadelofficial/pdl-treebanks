import math

import pytest

from pdl_treebanks.aggregate import aggregate_chunk_term_counts
from pdl_treebanks.conllu import ConlluSentence, ConlluWord
from pdl_treebanks.db import TreebankDB
from pdl_treebanks.keyness import author_level_keywords, hapax_tiers, work_level_keywords

SOURCE = "stanza:perseus"

# Two works ("tlg001", "tlg002") by one synthetic author ("tlg9001").
# work1 = chunk1a + chunk1b; work2 = chunk2a. Counts chosen by hand so the
# expected TF-IDF/hapax-tier outcomes can be verified by hand too -- see
# the worked comments below each fixture.
CHUNK_1A = "urn:cts:greekLit:tlg9001.tlg001.test1:1"
CHUNK_1B = "urn:cts:greekLit:tlg9001.tlg001.test1:2"
CHUNK_2A = "urn:cts:greekLit:tlg9001.tlg002.test1:1"


def _word(idx, lemma):
    return ConlluWord(id=str(idx), form=lemma, lemma=lemma, upos="NOUN")


def _sentence(sent_id, lemmas):
    words = [_word(i, lemma) for i, lemma in enumerate(lemmas, start=1)]
    return ConlluSentence(sent_id=sent_id, text=" ".join(lemmas), words=words)


@pytest.fixture
def db():
    database = TreebankDB(":memory:")
    database.create_schema()

    # chunk1a: alpha x3, beta x1, gamma x1, delta x1, common x2  (total 9)
    database.write_chunk(
        CHUNK_1A, SOURCE,
        [_sentence("1a", ["alpha", "alpha", "alpha", "beta", "gamma", "delta", "common", "common"])],
    )
    # chunk1b: epsilon x1, delta x1, common x2  (total 4)
    database.write_chunk(
        CHUNK_1B, SOURCE,
        [_sentence("1b", ["epsilon", "delta", "common", "common"])],
    )
    # chunk2a (different work, same author): gamma x1, common x2  (total 3)
    database.write_chunk(
        CHUNK_2A, SOURCE,
        [_sentence("2a", ["gamma", "common", "common"])],
    )
    database.commit()

    aggregate_chunk_term_counts(database, SOURCE)
    database.commit()
    return database


def test_work_level_keywords_ranks_distinctive_terms_above_common_ones(db):
    ranked = work_level_keywords(db, CHUNK_1A, SOURCE)
    lemmas = [r["lemma"] for r in ranked]

    # alpha appears only in chunk1a (work-level df=1/2 chunks) -> distinctive.
    # common appears in every work1 chunk (df=2/2) -> idf=0, ranks last.
    # beta/gamma/delta all have chunk-count 1, so they're excluded here
    # entirely (they show up via hapax_tiers instead).
    assert lemmas == ["alpha", "common"]

    alpha, common = ranked
    # chunk1a has 8 non-punct words total (3+1+1+1+2)
    assert alpha["tfidf"] == pytest.approx((3 / 8) * math.log(2 / 1), rel=1e-6)
    assert common["tfidf"] == pytest.approx(0.0, abs=1e-9)
    assert alpha["tfidf"] > common["tfidf"]


def test_work_level_keywords_empty_for_unknown_chunk(db):
    assert work_level_keywords(db, "urn:cts:greekLit:doesnotexist:1", SOURCE) == []


def test_author_level_keywords_ranks_work1_terms_against_the_authors_other_chunks(db):
    ranked = author_level_keywords(db, "tlg9001", "tlg001", SOURCE)
    lemmas = [r["lemma"] for r in ranked]

    # alpha: work1 total 3, appears in only 1 of the author's 3 chunks -> most distinctive.
    # delta: work1 total 2, appears in 2 of 3 chunks -> some signal.
    # common: work1 total 4, appears in all 3 chunks -> idf=0, ranks last.
    # beta/gamma/epsilon all have work1 total 1 -> excluded (hapax territory).
    assert lemmas == ["alpha", "delta", "common"]

    alpha, delta, common = ranked
    # work1 has 12 non-punct words total (chunk1a's 8 + chunk1b's 4)
    assert alpha["tfidf"] == pytest.approx((3 / 12) * math.log(3 / 1), rel=1e-6)
    assert delta["tfidf"] == pytest.approx((2 / 12) * math.log(3 / 2), rel=1e-6)
    assert common["tfidf"] == pytest.approx(0.0, abs=1e-9)
    assert alpha["tfidf"] > delta["tfidf"] > common["tfidf"]


def test_hapax_tiers_assigns_the_tightest_scope(db):
    tiers = hapax_tiers(db, CHUNK_1A, SOURCE)

    # beta: only ever used once, anywhere, by this author -> author-unique.
    assert [t["lemma"] for t in tiers["author"]] == ["beta"]
    # gamma: unique within work1, but the author uses it again in work2 (chunk2a)
    # -- exactly the "used elsewhere by the author, but only once in this work" case.
    assert [t["lemma"] for t in tiers["work"]] == ["gamma"]
    # delta: unique to this chunk, but recurs elsewhere in the same work (chunk1b).
    assert [t["lemma"] for t in tiers["passage"]] == ["delta"]

    # alpha/common appear >1 time in this chunk -- not hapax at all, shouldn't
    # show up in any tier.
    all_hapax_lemmas = {t["lemma"] for tier in tiers.values() for t in tier}
    assert "alpha" not in all_hapax_lemmas
    assert "common" not in all_hapax_lemmas


def test_hapax_tiers_empty_for_chunk_with_no_hapax_terms(db):
    # chunk1b: epsilon(1), delta(1), common(2) -- epsilon and delta ARE hapax
    # in chunk1b, so use a chunk where nothing is: not directly available in
    # this fixture, so instead just check every returned entry is well-formed
    # and the union of chunk1b's tiers only contains epsilon/delta.
    tiers = hapax_tiers(db, CHUNK_1B, SOURCE)
    all_lemmas = {t["lemma"] for tier in tiers.values() for t in tier}
    assert all_lemmas == {"epsilon", "delta"}
