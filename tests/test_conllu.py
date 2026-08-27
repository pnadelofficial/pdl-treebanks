import json
from pathlib import Path

from pdl_treebanks.conllu import sentences_from_nlp_pipeline_chunk

FIXTURE = Path(__file__).parent / "fixtures" / "iliad_two_sentences.json"


def _load_fixture():
    return json.loads(FIXTURE.read_text())


def test_splits_flat_tokens_into_sentences_on_id_reset():
    chunk = _load_fixture()

    sentences = sentences_from_nlp_pipeline_chunk(chunk, "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    assert len(sentences) == 2
    assert [w.form for w in sentences[0].words] == ["μῆνιν", "ἄειδε", "θεὰ", "."]
    assert [w.form for w in sentences[1].words] == ["Πηληϊάδεω", "δὲ", "Ἀχιλῆος", "ἦν", "."]


def test_word_ids_restart_per_sentence():
    chunk = _load_fixture()

    sentences = sentences_from_nlp_pipeline_chunk(chunk, "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    assert [w.id for w in sentences[0].words] == ["1", "2", "3", "4"]
    assert [w.id for w in sentences[1].words] == ["1", "2", "3", "4", "5"]


def test_head_and_deprel_are_preserved():
    chunk = _load_fixture()

    sentences = sentences_from_nlp_pipeline_chunk(chunk, "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    root = sentences[0].words[1]
    assert root.form == "ἄειδε"
    assert root.head == 0
    assert root.deprel == "root"


def test_punctuation_gets_no_urn_but_words_do():
    chunk = _load_fixture()

    sentences = sentences_from_nlp_pipeline_chunk(chunk, "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    period = sentences[0].words[-1]
    word = sentences[0].words[0]
    assert period.form == "."
    assert period.urn is None
    assert word.urn == "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1@μῆνιν[1]"


def test_to_text_renders_valid_conllu_shape():
    chunk = _load_fixture()
    sentences = sentences_from_nlp_pipeline_chunk(chunk, "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    text = sentences[0].to_text()
    lines = text.strip("\n").split("\n")

    assert lines[0].startswith("# sent_id = ")
    assert lines[1].startswith("# text = ")
    assert lines[2].split("\t")[:2] == ["1", "μῆνιν"]
    assert len(lines[2].split("\t")) == 10
