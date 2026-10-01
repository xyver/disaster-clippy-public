"""Export a standalone, consumer-only clippy_core package from this checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "__init__.py", "chat.py", "config.py", "context.py", "embeddings.py",
    "llm.py", "schemas.py", "prompts/default.md", "vectordb/__init__.py",
    "vectordb/base.py", "vectordb/sqlite_hybrid.py", "vectordb/pgvector.py",
)
PACKAGE_METADATA = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "clippy-core-runtime"
version = "0.2.0"
description = "Cited search and API-backed chat over prepared indexes"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
openai = ["openai>=1.30"]
anthropic = ["anthropic>=0.40"]
pgvector = ["asyncpg>=0.29", "psycopg2-binary>=2.9"]

[tool.setuptools.packages.find]
include = ["clippy_core*"]

[tool.setuptools.package-data]
clippy_core = ["prompts/*.md"]
"""
RUNTIME_README = """# clippy-core runtime

This folder is the consumer-only export of `clippy-core`. Supply a prepared
SQLite index, search it, and generate cited answers through a server-side
OpenAI or Anthropic API key. It contains no PDF builders, source tools,
evaluation commands, local LLM, or public API server.

Install with `pip install -e ".[openai]"` (or `[anthropic]`). Keep API keys in
your host backend, which should authenticate callers and enforce usage limits.

```python
from clippy_core import ChatService, ClippyConfig
from clippy_core.vectordb import SQLiteHybridStore

config = ClippyConfig(index_path="rules.sqlite", search_mode="keyword", llm_provider="openai")
store = SQLiteHybridStore(config.index_path, create=False, keyword_only=True)
answer = ChatService(store, config=config).chat_sync("What counts as a fall?")
print(answer.text, answer.search_results)
```

If the host already retrieved passages, map them to `SearchResult` and call
`ChatService(config=config).answer_sync(question, passages, max_evidence=5,
host_context="...")`. The returned `search_results` are the passages used to
build the current answer. The host authorizes passages and user context before
calling the runtime.

Keyword mode uses no local embedding model. Hybrid search needs the same
query embedding model that produced the prepared index.
"""


def export_runtime(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Export target already exists: {output}")
    package = output / "clippy_core"
    for relative in RUNTIME_FILES:
        source = ROOT / "clippy_core" / relative
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    (output / "pyproject.toml").write_text(PACKAGE_METADATA, encoding="utf-8")
    (output / "README.md").write_text(RUNTIME_README, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "runtime-only")
    args = parser.parse_args()
    print(export_runtime(args.output))
