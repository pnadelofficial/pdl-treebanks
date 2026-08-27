import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdl_treebanks.conllu import sentences_from_nlp_pipeline_chunk
from pdl_treebanks.db import TreebankDB

FIXTURE = Path(__file__).parent / "fixtures" / "iliad_two_sentences.json"
URN = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db = TreebankDB(db_path)
    db.create_schema()

    chunk = json.loads(FIXTURE.read_text())
    sentences = sentences_from_nlp_pipeline_chunk(chunk, URN)
    db.write_chunk(URN, "stanza:perseus", sentences)
    db.commit()
    db.close()

    monkeypatch.setenv("PDL_TREEBANKS_DB", str(db_path))
    import pdl_treebanks.server as server_module

    importlib.reload(server_module)

    return TestClient(server_module.app)


def test_get_treebank_json(client):
    response = client.get("/treebank", params={"urn": URN})

    assert response.status_code == 200
    sentences = response.json()
    assert len(sentences) == 2
    assert sentences[0]["words"][0]["form"] == "μῆνιν"
    assert sentences[0]["words"][1]["deprel"] == "root"


def test_get_treebank_conllu(client):
    response = client.get("/treebank", params={"urn": URN, "format": "conllu"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# sent_id = " in response.text
    assert "\tμῆνιν\t" in response.text


def test_get_treebank_missing_urn_404(client):
    response = client.get("/treebank", params={"urn": "urn:cts:greekLit:doesnotexist:1"})

    assert response.status_code == 404


def test_get_treebank_unknown_source_404(client):
    response = client.get("/treebank", params={"urn": URN, "source": "spacy:grc_proiel_trf"})

    assert response.status_code == 404
