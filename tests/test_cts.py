import pytest

from pdl_treebanks.cts import parse_cts_urn


def test_parses_full_urn_with_passage():
    parts = parse_cts_urn("urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1")

    assert parts.namespace == "greekLit"
    assert parts.text_group == "tlg0012"
    assert parts.work == "tlg001"
    assert parts.version == "perseus-grc2"
    assert parts.passage == "1.1"


def test_parses_latin_phi_urn():
    parts = parse_cts_urn("urn:cts:latinLit:phi0959.phi006.perseus-lat2:1.1")

    assert parts.text_group == "phi0959"
    assert parts.work == "phi006"


def test_version_is_none_when_absent():
    parts = parse_cts_urn("urn:cts:greekLit:tlg0012.tlg001:1.1")

    assert parts.text_group == "tlg0012"
    assert parts.work == "tlg001"
    assert parts.version is None


def test_passage_is_none_when_absent():
    parts = parse_cts_urn("urn:cts:greekLit:tlg0012.tlg001.perseus-grc2")

    assert parts.passage is None


@pytest.mark.parametrize(
    "urn",
    [
        "not a urn",
        "urn:cts:greekLit",
        "urn:cts:greekLit:tlg0012",
        "urn:notcts:greekLit:tlg0012.tlg001:1.1",
    ],
)
def test_raises_on_malformed_urn(urn):
    with pytest.raises(ValueError):
        parse_cts_urn(urn)
