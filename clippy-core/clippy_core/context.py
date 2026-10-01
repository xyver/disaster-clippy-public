"""
Context formatting: how search results are presented to the LLM.

The formatter numbers each passage so the model can cite ``[n]``, and puts
the citation (document, edition, section, page) and link right above the
text. Subclass ``ContextFormatter`` and override ``format_passage`` to change
the layout without touching ChatService.
"""

from __future__ import annotations

from typing import List

from .schemas import SearchResult


class ContextFormatter:
    def __init__(self, max_context_chars: int = 12000, per_result_chars: int = 2500):
        self.max_context_chars = max_context_chars
        self.per_result_chars = per_result_chars

    def format_passage(self, n: int, result: SearchResult) -> str:
        text = result.content.strip()
        if len(text) > self.per_result_chars:
            text = text[: self.per_result_chars].rsplit(" ", 1)[0] + " …"
        lines = [f"[{n}] {result.citation()}"]
        if result.url:
            lines.append(f"Link: {result.url}")
        lines.append('"""')
        lines.append(text)
        lines.append('"""')
        return "\n".join(lines)

    def format(self, results: List[SearchResult]) -> str:
        """Returns the context block. Stops adding passages at max_context_chars."""
        if not results:
            return "No relevant passages were found in the knowledge base."
        parts, total = [], 0
        for n, result in enumerate(results, 1):
            part = self.format_passage(n, result)
            if parts and total + len(part) > self.max_context_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n\n".join(parts)

    def used(self, results: List[SearchResult]) -> List[SearchResult]:
        """The results that ``format`` actually included (same order and numbering)."""
        out, total = [], 0
        for n, result in enumerate(results, 1):
            size = len(self.format_passage(n, result))
            if out and total + size > self.max_context_chars:
                break
            out.append(result)
            total += size
        return out


def extractive_answer(results: List[SearchResult], snippet_chars: int = 500) -> str:
    """Answer used when no LLM is configured: the top passages, cited."""
    if not results:
        return "No relevant passages were found in the knowledge base."
    lines = ["No LLM is configured, so here are the most relevant passages:", ""]
    for n, r in enumerate(results, 1):
        text = " ".join(r.content.split())
        if len(text) > snippet_chars:
            text = text[:snippet_chars].rsplit(" ", 1)[0] + " …"
        lines.append(f"[{n}] {r.citation()}")
        if r.url:
            lines.append(f"    {r.url}")
        lines.append(f"    {text}")
        lines.append("")
    return "\n".join(lines).rstrip()
