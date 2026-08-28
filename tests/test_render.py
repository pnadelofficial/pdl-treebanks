import json
from pathlib import Path

import lxml.html

from pdl_treebanks.conllu import ConlluSentence, ConlluWord, sentences_from_nlp_pipeline_chunk
from pdl_treebanks.render import render_sentence_outline

FIXTURE = Path(__file__).parent / "fixtures" / "iliad_two_sentences.json"
URN = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"


def _sentences():
    chunk = json.loads(FIXTURE.read_text())
    return sentences_from_nlp_pipeline_chunk(chunk, URN)


def _parse(outline_html: str):
    # The output is real HTML (e.g. bare `open` on <details>), not strict
    # XML -- lxml.html is a permissive HTML parser, unlike
    # xml.etree.ElementTree, so it round-trips it without complaint.
    return lxml.html.fromstring(outline_html)


def test_renders_well_formed_html_fragment():
    root = _parse(render_sentence_outline(_sentences()[0]))
    assert root.tag == "ul"


def test_every_word_appears_exactly_once():
    sentence = _sentences()[0]  # "μῆνιν ἄειδε θεὰ." -- 4 words incl. punctuation
    root = _parse(render_sentence_outline(sentence))

    words = [el.text for el in root.iter("span") if el.get("class") == "ol-word"]
    assert sorted(words) == sorted(w.form for w in sentence.words)


def test_root_word_is_present_and_tagged_root():
    sentence = _sentences()[0]
    outline = render_sentence_outline(sentence)
    assert "ἄειδε" in outline  # the root verb

    root = _parse(outline)
    tags = [el.text for el in root.iter("span") if el.get("class") == "ol-tag"]
    assert any("root" in (t or "") for t in tags)


def test_leaf_words_are_not_wrapped_in_details():
    # In "μῆνιν ἄειδε θεὰ." the punctuation "." has no dependents, so its
    # <li> should be a bare leaf, not a <details>.
    sentence = _sentences()[0]
    root = _parse(render_sentence_outline(sentence))

    leaf_lis = [li for li in root.iter("li") if li.find("details") is None]
    assert len(leaf_lis) >= 1
    for li in leaf_lis:
        assert li.find("span") is not None


def test_word_with_no_head_resolving_in_sentence_becomes_its_own_root():
    # Defensive case: a word whose head isn't in this sentence's word set
    # (see render.py module docstring) shouldn't be dropped or crash --
    # it should render as an additional top-level entry.
    sentence = ConlluSentence(
        sent_id="s1",
        text="a b",
        words=[
            ConlluWord(id="1", form="a", upos="NOUN", head=99, deprel="obj"),  # 99 doesn't exist here
            ConlluWord(id="2", form="b", upos="VERB", head=0, deprel="root"),
        ],
    )
    root = _parse(render_sentence_outline(sentence))

    assert len(root) == 2  # both "a" and "b" render as top-level <li>s
    words = {el.text for el in root.iter("span") if el.get("class") == "ol-word"}
    assert words == {"a", "b"}


def test_empty_sentence_does_not_crash():
    sentence = ConlluSentence(sent_id="s1", text="", words=[])
    outline = render_sentence_outline(sentence)
    _parse(outline)  # just shouldn't raise


def test_handles_a_word_with_ampersand_safely():
    sentence = ConlluSentence(
        sent_id="s1",
        text="a & b",
        words=[ConlluWord(id="1", form="a & b", upos="NOUN", head=0, deprel="root")],
    )
    root = _parse(render_sentence_outline(sentence))
    assert any(el.text == "a & b" for el in root.iter("span") if el.get("class") == "ol-word")
