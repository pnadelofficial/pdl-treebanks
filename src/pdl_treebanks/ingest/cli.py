"""Ingest MinimumViablePerseus proto-pages into a pdl-treebanks SQLite db.

Mirrors MinimumViablePerseus's src/tools/run_tokenizer.py: walks
--proto-dir's index.json files, extracts each chunk's primary text via
kodon_py's TEIParser, and calls nlp_pipeline over HTTP. Chunks whose
(cts_urn, source) is already in --db are skipped unless --force.

Usage:
    pdl-treebanks-ingest --proto-dir ../MinimumViablePerseus/proto-pages \\
        --db treebanks.db --nlp-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from lxml import etree

from pdl_treebanks.db import TreebankDB
from pdl_treebanks.ingest.nlp_pipeline_source import SOURCE_NAME, analyze_chunk


def _iter_chunk_files(proto_dir: Path):
    for index_file in sorted(proto_dir.glob("**/index.json")):
        version_dir = index_file.parent
        chunks = json.load(open(index_file)).get("chunks", [])
        for entry in chunks:
            chunk_file = version_dir / entry["file"]
            if chunk_file.exists():
                yield chunk_file


def _primary_text(chunk_file: Path) -> tuple[str, str]:
    """Return (cts_urn, primary_text) for a compiled chunk XML file. Kept
    a local, self-contained copy of MinimumViablePerseus's own
    _primary_text (run_tokenizer.py) rather than importing the mvp
    package, to keep pdl-treebanks decoupled from mvp's app internals --
    kodon_py, the actual TEI parser, is a normal external dependency
    either way."""
    from kodon_py.tei_parser import TEIParser

    root = etree.parse(chunk_file).getroot()
    cts_urn = root.get("cts_urn", "")
    base_urn = root.get("base_urn", "") or cts_urn.rsplit(":", 1)[0]
    chunk_unit = root.get("unit", "")

    content_el = root.find("elements")
    if content_el is None:
        raise ValueError(f"No <elements> in {chunk_file}")

    parser = TEIParser(content_el, base_urn, chunk_unit)
    return cts_urn, parser.primary_text


def _already_ingested(db: TreebankDB, cts_urn: str, source: str) -> bool:
    return db.read_chunk(cts_urn, source) is not None


def run(proto_dir: Path, db_path: Path, nlp_url: str, lang: str | None, force: bool) -> None:
    nlp_url = nlp_url.rstrip("/")
    db = TreebankDB(db_path)
    db.create_schema()

    generated = skipped = failed = 0

    with httpx.Client(timeout=60.0) as client:
        for chunk_file in _iter_chunk_files(proto_dir):
            try:
                cts_urn, text = _primary_text(chunk_file)
            except (etree.XMLSyntaxError, ValueError) as exc:
                print(f"  FAILED (parse): {chunk_file}: {exc}", file=sys.stderr)
                failed += 1
                continue

            if not text.strip():
                skipped += 1
                continue

            if not force and _already_ingested(db, cts_urn, SOURCE_NAME):
                skipped += 1
                continue

            try:
                sentences = analyze_chunk(client, nlp_url, cts_urn, text, lang=lang)
            except httpx.HTTPError as exc:
                print(f"  FAILED (nlp_pipeline): {cts_urn}: {exc}", file=sys.stderr)
                failed += 1
                continue

            try:
                db.write_chunk(cts_urn, SOURCE_NAME, sentences)
            except ValueError as exc:
                print(f"  FAILED (urn): {cts_urn}: {exc}", file=sys.stderr)
                failed += 1
                continue

            db.commit()
            generated += 1

    db.close()
    print(f"Treebanks: {generated} generated, {skipped} skipped, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest proto-pages into a pdl-treebanks db")
    parser.add_argument("--proto-dir", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--nlp-url", required=True)
    parser.add_argument(
        "--lang",
        default=None,
        help="Explicit language code (e.g. grc, la) for every chunk ingested "
        "this run, bypassing nlp_pipeline's own language detection. Omit to "
        "let nlp_pipeline detect it per chunk.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run(args.proto_dir.resolve(), args.db, args.nlp_url, args.lang, args.force)


if __name__ == "__main__":
    main()
