"""HTTP API serving prebuilt treebank data from a TreebankDB.

Shape follows pdl-morph-server: ingestion is a separate offline step
(pdl_treebanks.ingest.cli) that fills a SQLite db; this process only reads
it. PORT/WEB_CONCURRENCY env vars match pdl-morph-server's own convention.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from pdl_treebanks.db import TreebankDB
from pdl_treebanks.ingest.nlp_pipeline_source import SOURCE_NAME

DB_PATH = os.environ.get("PDL_TREEBANKS_DB", "treebanks.db")

app = FastAPI(title="pdl-treebanks")
db = TreebankDB(DB_PATH)


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
