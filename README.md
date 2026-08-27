# pdl-treebanks

HTTP API serving prebuilt treebank (dependency-parse) data for Perseus Digital
Library texts. Canonical internal shape is CoNLL-U; the first source is
Stanza via [nlp_pipeline](https://github.com/PerseusDLCode/nlp_pipeline),
converted at ingest time (`pdl_treebanks.ingest`). Serving is a thin FastAPI
app reading a prebuilt SQLite db (`pdl_treebanks.db`) -- the same
ingest-offline / serve-live split as
[pdl-morph-server](https://github.com/PerseusDLCode/pdl-morph-server), and
the source-per-chunk schema is meant to make adding a second source (spaCy,
expert annotation) additive rather than a rewrite.

## Setup

```sh
uv sync
```

## Ingesting

Requires a running `nlp_pipeline` server (see that repo's README) and a
local checkout of [MinimumViablePerseus](https://github.com/PerseusDLCode/MinimumViablePerseus)
for its `proto-pages`:

```sh
uv run pdl-treebanks-ingest \
    --proto-dir ../MinimumViablePerseus/proto-pages \
    --db treebanks.db \
    --nlp-url http://localhost:8000
```

## Aggregating

A separate offline step (mirroring pdl-morph-server's own load/aggregate/ingest
split) computes per-chunk term (lemma, morphology) counts, the basis for
work- and author-level keyness rankings. Run after ingesting (or re-run
after ingesting more):

```sh
uv run pdl-treebanks-aggregate --db treebanks.db --source stanza:perseus
```

## Running

Dev server (autoreload, binds to `127.0.0.1`):

```sh
uv run pdl-treebanks-dev
```

Production server (binds to `0.0.0.0`, multiple workers):

```sh
uv run pdl-treebanks
```

Both respect `PORT` (default `8000`) and `PDL_TREEBANKS_DB` (default
`treebanks.db`); the production server also respects `WEB_CONCURRENCY`
(default `4`).

## API

`GET /treebank?urn=<cts_urn>&source=<source>&format=json|conllu`

Returns the sentences (and per-word CoNLL-U fields) for the given chunk CTS
URN and source (default source: `stanza:perseus`). `format=conllu` returns
raw CoNLL-U text instead of JSON.

## Tests

```sh
uv run pytest
```
