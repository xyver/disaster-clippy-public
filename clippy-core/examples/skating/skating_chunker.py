"""
SkatingRuleChunker - starting point for the skating rulebooks.

It is a RegexSectionChunker (one chunk per rule, long rules split into parts)
plus two skating-specific additions you are expected to extend:

  - ``disciplines``: map rule-number ranges to a discipline, so searches can
    be filtered with  -f discipline=singles
  - ``doc_type = "rules"`` on every chunk

Ideas for the domain team (see HANDOFF.md):
  - cite paragraphs, not just rules: split each rule on "1.", "2." ... and set
    section_id = "Rule 611.2" so answers can say exactly which paragraph
  - attach rule updates / technical panel handbooks as separate documents
    with their own doc_id, and link them to the rules they modify
  - normalise element names (e.g. "3Lz" vs "triple Lutz") into a synonyms
    field so keyword search finds both
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from clippy_core.ingest.chunkers import RegexSectionChunker
from clippy_core.ingest.documents import Document
from clippy_core.schemas import Chunk


class SkatingRuleChunker(RegexSectionChunker):
    name = "skating_rule"

    def __init__(self, disciplines: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.ranges = []
        for discipline, span in (disciplines or {}).items():
            lo, hi = (int(x) for x in str(span).split("-", 1))
            self.ranges.append((lo, hi, discipline))

    def discipline_for(self, section_id: str) -> Optional[str]:
        m = re.search(r"\d+", section_id or "")
        if not m:
            return None
        n = int(m.group())
        return next((d for lo, hi, d in self.ranges if lo <= n <= hi), None)

    def chunk(self, doc: Document) -> List[Chunk]:
        chunks = super().chunk(doc)
        for ch in chunks:
            ch.metadata["doc_type"] = "rules"
            discipline = self.discipline_for(ch.metadata.get("section_id", ""))
            if discipline:
                ch.metadata["discipline"] = discipline
        return chunks
