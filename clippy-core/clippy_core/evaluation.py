"""
Retrieval evaluation against a golden question set.

golden.yaml:

    k: 5                                  # how deep to look (default 5)
    filters: {edition: "2025-26"}         # optional default filters
    questions:
      - q: How many jump elements are allowed in the senior short program?
        expect: {section_id: "Rule 611"}  # a result matching ALL these metadata keys is a hit
      - q: What is a Rittberger?
        expect: [{section_id: "Rule 610"}, {section_id: "Rule 612"}]   # any of these
      - q: Maximum program length for juniors
        expect_text: "2 minutes 40 seconds"   # or: a result whose text contains this
        filters: {edition: "2024-25"}      # per-question override

Metrics:
    hit@1   share of questions whose top result is a hit
    hit@k   share with a hit anywhere in the top k
    MRR     mean reciprocal rank of the first hit (1.0 = always first)

Run it after every chunker, prompt or embedding change:
    clippy eval golden.yaml --index build/rules.sqlite --compare
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .schemas import SearchResult


def load_golden(path: Union[str, Path]) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not data.get("questions"):
        raise ValueError(f"{path}: no 'questions' found")
    return data


def _matches(result: SearchResult, expect: Any, expect_text: Optional[str]) -> bool:
    if expect_text and expect_text.lower() in result.content.lower():
        return True
    if not expect:
        return False
    options = expect if isinstance(expect, list) else [expect]
    meta = {**result.metadata, "id": result.id, "source_id": result.source_id}
    return any(all(str(meta.get(k)) == str(v) for k, v in opt.items()) for opt in options)


@dataclass
class QuestionResult:
    question: str
    rank: Optional[int]            # 1-based rank of first hit, None if missed
    top: List[str] = field(default_factory=list)   # citations of the top results


@dataclass
class EvalReport:
    mode: str
    k: int
    results: List[QuestionResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def hit_at_1(self) -> float:
        return sum(1 for r in self.results if r.rank == 1) / self.n if self.n else 0.0

    @property
    def hit_at_k(self) -> float:
        return sum(1 for r in self.results if r.rank) / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return sum(1.0 / r.rank for r in self.results if r.rank) / self.n if self.n else 0.0

    def summary(self, show_misses: bool = True) -> str:
        lines = [f"[{self.mode}] {self.n} questions  hit@1 {self.hit_at_1:.0%}  "
                 f"hit@{self.k} {self.hit_at_k:.0%}  MRR {self.mrr:.2f}"]
        if show_misses:
            for r in self.results:
                if not r.rank:
                    lines.append(f"  MISS: {r.question}")
                    lines += [f"        got: {c}" for c in r.top[:3]]
        return "\n".join(lines)


def evaluate(store, golden: Dict[str, Any], mode: str = "hybrid", k: Optional[int] = None) -> EvalReport:
    k = int(k or golden.get("k", 5))
    default_filters = golden.get("filters") or None
    out: List[QuestionResult] = []
    for item in golden["questions"]:
        results = store.search(item["q"], n_results=k, filters=item.get("filters", default_filters),
                               mode=mode)
        rank = next((i for i, r in enumerate(results, 1)
                     if _matches(r, item.get("expect"), item.get("expect_text"))), None)
        out.append(QuestionResult(item["q"], rank, [r.citation() for r in results]))
    return EvalReport(mode, k, out)
