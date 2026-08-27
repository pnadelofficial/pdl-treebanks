import json
from pathlib import Path

import pytest

from pdl_treebanks.conllu import sentences_from_nlp_pipeline_chunk
from pdl_treebanks.db import TreebankDB

FIXTURE = Path(__file__).parent / "fixtures" / "iliad_two_sentences.json"
URN = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"


def _fixture_sentences():
    chunk = json.loads(FIXTURE.read_text())
    return sentences_from_nlp_pipeline_chunk(chunk, URN)


def test_write_then_read_round_trips():
    db = TreebankDB(":memory:")
    db.create_schema()

    sentences = _fixture_sentences()
    db.write_chunk(URN, "stanza:perseus", sentences)
    db.commit()

    read_back = db.read_chunk(URN, "stanza:perseus")

    assert len(read_back) == len(sentences)
    assert [w.form for w in read_back[0].words] == [w.form for w in sentences[0].words]
    assert [w.head for w in read_back[0].words] == [w.head for w in sentences[0].words]
    assert [w.deprel for w in read_back[0].words] == [w.deprel for w in sentences[0].words]
    assert read_back[0].words[0].urn == sentences[0].words[0].urn


def test_read_missing_chunk_returns_none():
    db = TreebankDB(":memory:")
    db.create_schema()

    assert db.read_chunk("urn:cts:greekLit:doesnotexist:1", "stanza:perseus") is None


def test_write_chunk_replaces_existing():
    db = TreebankDB(":memory:")
    db.create_schema()

    sentences = _fixture_sentences()
    db.write_chunk(URN, "stanza:perseus", sentences)
    db.commit()
    db.write_chunk(URN, "stanza:perseus", sentences[:1])
    db.commit()

    read_back = db.read_chunk(URN, "stanza:perseus")
    assert len(read_back) == 1


def test_different_sources_coexist():
    db = TreebankDB(":memory:")
    db.create_schema()

    sentences = _fixture_sentences()
    db.write_chunk(URN, "stanza:perseus", sentences)
    db.write_chunk(URN, "expert:agldt", sentences[:1])
    db.commit()

    assert len(db.read_chunk(URN, "stanza:perseus")) == 2
    assert len(db.read_chunk(URN, "expert:agldt")) == 1


def test_chunks_for_work_and_text_group():
    db = TreebankDB(":memory:")
    db.create_schema()

    sentences = _fixture_sentences()
    iliad_1_1 = URN
    iliad_1_33 = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.33"
    odyssey_1_1 = "urn:cts:greekLit:tlg0012.tlg002.perseus-grc2:1.1"
    other_author = "urn:cts:greekLit:tlg0026.tlg003.perseus-grc2:1"

    for urn in (iliad_1_1, iliad_1_33, odyssey_1_1, other_author):
        db.write_chunk(urn, "stanza:perseus", sentences)
    db.commit()

    assert db.chunks_for_work("tlg0012", "tlg001", "stanza:perseus") == [iliad_1_1, iliad_1_33]
    assert set(db.chunks_for_text_group("tlg0012", "stanza:perseus")) == {
        iliad_1_1,
        iliad_1_33,
        odyssey_1_1,
    }
    assert db.chunks_for_work("tlg0012", "tlg999", "stanza:perseus") == []


def test_write_chunk_raises_on_malformed_urn():
    db = TreebankDB(":memory:")
    db.create_schema()

    with pytest.raises(ValueError):
        db.write_chunk("not-a-cts-urn", "stanza:perseus", _fixture_sentences())
