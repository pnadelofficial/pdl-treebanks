"""Dependency-tree rendering: a plain nested, collapsible text outline.

Replaces an earlier SVG arc/tree-diagram approach (still visible in
mockups/dependency-tree.html as a historical artifact) that turned out too
cluttered for real sentences -- deprel labels on every edge, crossing arcs,
and pan/zoom chrome that risked re-inventing displaCy's own problems.
This is deliberately much simpler: one <details>/<summary> per word with
children, using the browser's native disclosure widget for collapsing --
no layout math, no SVG, no JS at all.

Any word whose head doesn't resolve to another word in the same sentence
(defensively handled, not assumed impossible) is rendered as its own
top-level entry rather than being dropped or drawn as a dangling edge.
"""

from __future__ import annotations

from pdl_treebanks.conllu import ConlluSentence


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tag(word) -> str:
    bits = [b for b in (word.upos, word.deprel) if b]
    return " · ".join(_escape(b) for b in bits)


def _title(word) -> str:
    bits = [b for b in (word.lemma, word.feats) if b]
    return _escape(" | ".join(bits))


def _render_node(word_id: int, by_id: dict, children: dict[int, list[int]]) -> str:
    word = by_id[word_id]
    label = (
        f'<span class="ol-word" title="{_title(word)}">{_escape(word.form)}</span> '
        f'<span class="ol-tag">{_tag(word)}</span>'
    )
    kid_ids = sorted(children.get(word_id, []))
    if not kid_ids:
        # No count badge and no disclosure marker at all (see CSS) -- a
        # leaf should visibly have nothing to click, not just a subtle
        # marker easy to miss next to ones that do.
        return f'<li class="ol-leaf">{label}</li>'
    count_word = "dependent" if len(kid_ids) == 1 else "dependents"
    badge = f'<span class="ol-count">{len(kid_ids)} {count_word}</span>'
    inner = "".join(_render_node(k, by_id, children) for k in kid_ids)
    return f"<li><details open><summary>{label} {badge}</summary><ul>{inner}</ul></details></li>"


def render_sentence_outline(sentence: ConlluSentence) -> str:
    """Render one sentence's dependency structure as a nested, collapsible
    <ul>/<details> outline -- one top-level <li> per root (normally just
    one; more than one signals a word whose head fell outside this
    sentence, see module docstring)."""
    # Multiword-token range lines (id like "2-3") aren't nodes in the
    # dependency graph -- only their expanded words are.
    by_id = {int(w.id): w for w in sentence.words if "-" not in w.id}
    if not by_id:
        return '<ul class="tree-outline"><li class="ol-empty">(no words)</li></ul>'

    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for word_id, word in by_id.items():
        if word.head and word.head in by_id and word.head != word_id:
            children.setdefault(word.head, []).append(word_id)
        else:
            roots.append(word_id)

    items = "".join(_render_node(r, by_id, children) for r in sorted(roots))
    return f'<ul class="tree-outline">{items}</ul>'
