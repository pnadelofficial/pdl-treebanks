"""Minimal CTS URN parsing.

Just enough to key treebank chunks by author (text_group) and work for
indexed author/work-level lookups, without pulling in a full CTS
resolution library (perseus-cts) for what's plain string parsing here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CtsUrnParts:
    namespace: str
    text_group: str
    work: str
    version: str | None
    passage: str | None


def parse_cts_urn(urn: str) -> CtsUrnParts:
    """Parse `urn:cts:{namespace}:{text_group}.{work}[.{version}]:[{passage}]`.

    e.g. urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:1.1 ->
    namespace=greekLit, text_group=tlg0012, work=tlg001,
    version=perseus-grc2, passage=1.1

    Raises ValueError if the URN doesn't have at least a namespace,
    text_group, and work -- those are required for author/work-level
    lookups, so a chunk with an unparseable URN should fail loudly at
    ingest time rather than being silently stored with a null
    text_group/work that would just never show up in those lookups.
    """
    parts = urn.split(":")
    if len(parts) < 4 or parts[0] != "urn" or parts[1] != "cts":
        raise ValueError(f"Not a CTS URN: {urn!r}")

    namespace = parts[2]
    work_component = parts[3]
    passage = parts[4] if len(parts) > 4 and parts[4] else None

    segments = work_component.split(".")
    if len(segments) < 2 or not segments[0] or not segments[1]:
        raise ValueError(f"CTS URN work component missing text_group/work: {urn!r}")

    text_group, work = segments[0], segments[1]
    version = segments[2] if len(segments) > 2 and segments[2] else None

    return CtsUrnParts(
        namespace=namespace,
        text_group=text_group,
        work=work,
        version=version,
        passage=passage,
    )
