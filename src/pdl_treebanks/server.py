"""HTTP API serving prebuilt treebank data from a TreebankDB.

Shape follows pdl-morph-server: ingestion is a separate offline step
(pdl_treebanks.ingest.cli) that fills a SQLite db; this process only reads
it. PORT/WEB_CONCURRENCY env vars match pdl-morph-server's own convention.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pdl_treebanks.cts import parse_cts_urn
from pdl_treebanks.db import TreebankDB
from pdl_treebanks.ingest.nlp_pipeline_source import SOURCE_NAME
from pdl_treebanks.keyness import author_level_keywords, hapax_tiers, work_level_keywords
from pdl_treebanks.render import render_sentence_outline

DB_PATH = os.environ.get("PDL_TREEBANKS_DB", "treebanks.db")
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="pdl-treebanks")
db = TreebankDB(DB_PATH)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class WordOut(BaseModel):
    id: str
    form: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: str | None
    head: int | None
    deprel: str | None
    deps: str | None
    misc: dict[str, str]
    urn: str | None


class SentenceOut(BaseModel):
    sent_id: str
    text: str
    words: list[WordOut]


@app.get("/")
def root():
    return {"message": "heartbeat"}


@app.get("/treebank", response_model=list[SentenceOut])
def get_treebank(
    urn: str = Query(..., description="CTS URN of the chunk, e.g. urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1"),
    source: str = Query(SOURCE_NAME, description="Annotation source name"),
    format: str = Query("json", pattern="^(json|conllu)$"),
):
    sentences = db.read_chunk(urn, source)
    if sentences is None:
        raise HTTPException(status_code=404, detail=f"No treebank for {urn!r} from source {source!r}")

    if format == "conllu":
        text = "\n".join(s.to_text() for s in sentences)
        return PlainTextResponse(content=text)

    return [
        SentenceOut(
            sent_id=s.sent_id,
            text=s.text,
            words=[
                WordOut(
                    id=w.id,
                    form=w.form,
                    lemma=w.lemma,
                    upos=w.upos,
                    xpos=w.xpos,
                    feats=w.feats,
                    head=w.head,
                    deprel=w.deprel,
                    deps=w.deps,
                    misc=w.misc,
                    urn=w.urn,
                )
                for w in s.words
            ],
        )
        for s in sentences
    ]


@app.get("/treebank/keywords")
def get_treebank_keywords(
    urn: str = Query(..., description="CTS URN of the chunk"),
    source: str = Query(SOURCE_NAME, description="Annotation source name"),
):
    """Keyness data for a chunk: work-level and author-level ranked terms
    (see pdl_treebanks.keyness), plus the three-tier hapax breakdown. This
    is the data the eventual reading-page key-word dashboard would render;
    no UI here, just the JSON."""
    if db.read_chunk(urn, source) is None:
        raise HTTPException(status_code=404, detail=f"No treebank for {urn!r} from source {source!r}")

    urn_parts = parse_cts_urn(urn)
    return {
        "text_group": urn_parts.text_group,
        "work": urn_parts.work,
        "work_keywords": work_level_keywords(db, urn, source),
        "author_keywords": author_level_keywords(db, urn_parts.text_group, urn_parts.work, source),
        "hapax_tiers": hapax_tiers(db, urn, source),
    }


@app.get("/treebank/view")
def get_treebank_view(
    request: Request,
    urn: str = Query(..., description="CTS URN of the chunk"),
    source: str = Query(SOURCE_NAME, description="Annotation source name"),
):
    """Rendered dependency-tree page for every sentence in a chunk, plus
    the keyness data (see get_treebank_keywords) shown as plain tables --
    this is the "external viewer" MinimumViablePerseus's reading page would
    link out to, one link per chunk."""
    sentences = db.read_chunk(urn, source)
    if sentences is None:
        raise HTTPException(status_code=404, detail=f"No treebank for {urn!r} from source {source!r}")

    urn_parts = parse_cts_urn(urn)
    return templates.TemplateResponse(
        request,
        "treebank_view.html.jinja",
        {
            "cts_urn": urn,
            "source": source,
            "text_group": urn_parts.text_group,
            "sentences": [{"sentence": s, "outline": render_sentence_outline(s)} for s in sentences],
            "work_keywords": work_level_keywords(db, urn, source),
            "author_keywords": author_level_keywords(db, urn_parts.text_group, urn_parts.work, source),
            "hapax": hapax_tiers(db, urn, source),
        },
    )


def main_dev() -> None:
    uvicorn.run("pdl_treebanks.server:app", host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), reload=True)


def main() -> None:
    uvicorn.run(
        "pdl_treebanks.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=int(os.environ.get("WEB_CONCURRENCY", 4)),
    )


if __name__ == "__main__":
    main_dev()
