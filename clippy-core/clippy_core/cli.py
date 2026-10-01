"""
clippy - command line for clippy_core.

    clippy build --manifest sources.yaml [--rebuild]     PDFs -> index
    clippy build docs/*.pdf --index docs.sqlite          quick build, generic chunker
    clippy preview --manifest sources.yaml [--doc ID]    show chunks without embedding
    clippy info --manifest sources.yaml                  what's in the index
    clippy search "query" --manifest sources.yaml        ranked passages with citations
    clippy ask "question" --manifest sources.yaml        cited answer
    clippy eval golden.yaml --manifest sources.yaml --compare
    clippy serve --manifest sources.yaml                 local web UI + JSON API

Every command accepts --index PATH instead of --manifest. Filters:
    --filter edition=2025-26 --filter discipline=singles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ClippyConfig


def _filters(pairs: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not pairs:
        return None
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--filter expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out.setdefault(key, []).append(value)
    return {k: v[0] if len(v) == 1 else v for k, v in out.items()}


def _config(args) -> ClippyConfig:
    config = ClippyConfig.from_env()
    if getattr(args, "manifest", None):
        from .ingest.pipeline import load_manifest
        manifest = load_manifest(args.manifest)
        config = ClippyConfig.from_dict(manifest.get("config") or {}, base=config)
        if manifest.get("index"):
            config.index_path = manifest["index"]
    if getattr(args, "index", None):
        config.index_path = args.index
    if getattr(args, "provider", None):
        config.llm_provider = args.provider
    return config


def _store(config: ClippyConfig, search_mode: Optional[str] = None):
    from .vectordb import SQLiteHybridStore
    path = Path(config.index_path)
    if not path.exists():
        raise SystemExit(f"No index at {path}. Run `clippy build` first.")
    return SQLiteHybridStore(path, api_key=config.get_openai_api_key(), create=False,
                             keyword_only=(search_mode or config.search_mode) == "keyword")


# ---------------------------------------------------------------- commands

def cmd_build(args) -> None:
    from .ingest.pipeline import build_index, load_manifest, manifest_from_files
    if args.manifest:
        manifest = load_manifest(args.manifest)
        if args.index:
            manifest["index"] = str(Path(args.index).resolve())
    elif args.files:
        chunker = json.loads(args.chunker) if args.chunker and args.chunker.startswith("{") else args.chunker
        manifest = manifest_from_files(args.files, source_id=args.source_id,
                                       index=args.index or "index.sqlite", chunker=chunker)
    else:
        raise SystemExit("Give --manifest sources.yaml or one or more PDF files")
    report = build_index(manifest, rebuild=args.rebuild, progress=print)
    print()
    print(report.summary())


def cmd_preview(args) -> None:
    from .ingest.chunkers import load_chunker
    from .ingest.pdf import extract_pages
    from .ingest.pipeline import _document, load_manifest, manifest_from_files, parse_page_ranges

    if args.manifest:
        manifest = load_manifest(args.manifest)
    elif args.files:
        chunker = json.loads(args.chunker) if args.chunker and args.chunker.startswith("{") else args.chunker
        manifest = manifest_from_files(args.files, chunker=chunker)
    else:
        raise SystemExit("Give --manifest sources.yaml or a PDF file")
    for entry in manifest["documents"]:
        doc = _document(entry, manifest.get("defaults") or {}, Path(manifest["_base"]))
        if args.doc and doc.doc_id != args.doc:
            continue
        chunker = load_chunker(entry.get("chunker") or manifest.get("chunker"), Path(manifest["_base"]))
        doc.pages = extract_pages(doc.path)
        keep = parse_page_ranges(entry.get("pages"), len(doc.pages))
        doc.pages = [p for p in doc.pages if p.num in keep]
        chunks = chunker.chunk(doc)
        print(f"== {doc.doc_id} {doc.edition}: {len(doc.pages)} pages -> {len(chunks)} chunks ({chunker.name})")
        for ch in chunks[: args.limit]:
            m = ch.metadata
            label = " · ".join(str(x) for x in (m.get("section_id"), m.get("section_title")) if x)
            pages = m["page_start"] if m["page_start"] == m["page_end"] else f"{m['page_start']}-{m['page_end']}"
            print(f"\n[{ch.id}] p.{pages}  {label}  ({len(ch.content)} chars)")
            text = ch.content if args.full else ch.content[:300] + ("…" if len(ch.content) > 300 else "")
            print("  " + text.replace("\n", "\n  "))
        if len(chunks) > args.limit:
            print(f"\n… {len(chunks) - args.limit} more (use --limit)")


def cmd_info(args) -> None:
    store = _store(_config(args))
    print(json.dumps(store.info(), indent=2))


def cmd_search(args) -> None:
    config = _config(args)
    store = _store(config, search_mode=args.mode)
    results = store.search(args.query, n_results=args.k, filters=_filters(args.filter),
                           mode=args.mode or config.search_mode)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        return
    if not results:
        print("No results.")
    for n, r in enumerate(results, 1):
        text = " ".join(r.content.split())
        print(f"[{n}] {r.citation()}   (score {r.score:.4f})")
        if r.url:
            print(f"    {r.url}")
        print(f"    {text[:400]}{'…' if len(text) > 400 else ''}\n")


def cmd_ask(args) -> None:
    from .chat import ChatService
    config = _config(args)
    chat = ChatService(_store(config), config=config)
    resp = chat.chat_sync(args.question, filters=_filters(args.filter))
    if args.json:
        print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))
        return
    print(resp.text)
    if resp.method.value != "simple" and resp.search_results:
        print("\nSources:")
        for n, r in enumerate(resp.search_results, 1):
            print(f"  [{n}] {r.citation()}" + (f"  {r.url}" if r.url else ""))
    if resp.error:
        print(f"\n({resp.error})", file=sys.stderr)


def cmd_eval(args) -> None:
    from .evaluation import evaluate, load_golden
    golden = load_golden(args.golden)
    store = _store(_config(args))
    modes = ["keyword", "semantic", "hybrid"] if args.compare else [args.mode or "hybrid"]
    for mode in modes:
        print(evaluate(store, golden, mode=mode, k=args.k).summary(show_misses=not args.compare or mode == "hybrid"))


def cmd_serve(args) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Install the server extra: pip install 'clippy-core[server]'")
    from .server import create_app
    config = _config(args)
    app = create_app(config, _store(config))
    print(f"Serving {config.index_path} at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clippy", description="Cited search and chat over your documents.")
    sub = p.add_subparsers(dest="command", required=True)

    def target(sp):
        sp.add_argument("--manifest", "-m", help="sources.yaml")
        sp.add_argument("--index", "-i", help="index .sqlite path (overrides manifest)")

    sp = sub.add_parser("build", help="build an index from PDFs")
    target(sp)
    sp.add_argument("files", nargs="*", help="PDF files (instead of --manifest)")
    sp.add_argument("--rebuild", action="store_true", help="delete the index first")
    sp.add_argument("--chunker", help="chunker spec for plain files, e.g. page or '{\"class\":...}'")
    sp.add_argument("--source-id", default="docs")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("preview", help="show how documents will be chunked (no embedding)")
    target(sp)
    sp.add_argument("files", nargs="*")
    sp.add_argument("--chunker")
    sp.add_argument("--doc", help="only this doc_id")
    sp.add_argument("--limit", type=int, default=15)
    sp.add_argument("--full", action="store_true", help="print full chunk text")
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("info", help="describe an index")
    target(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("search", help="ranked passages with citations")
    target(sp)
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--mode", choices=["hybrid", "keyword", "semantic"])
    sp.add_argument("--filter", "-f", action="append", help="key=value (repeatable)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="cited answer to a question")
    target(sp)
    sp.add_argument("question")
    sp.add_argument("--filter", "-f", action="append")
    sp.add_argument("--provider", choices=["auto", "anthropic", "openai", "none"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="score retrieval against a golden question set")
    target(sp)
    sp.add_argument("golden")
    sp.add_argument("-k", type=int)
    sp.add_argument("--mode", choices=["hybrid", "keyword", "semantic"])
    sp.add_argument("--compare", action="store_true", help="run all three modes")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("serve", help="local web UI and JSON API")
    target(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--provider", choices=["auto", "anthropic", "openai", "none"])
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:      # e.g. `clippy info | head`
        sys.stderr.close()


if __name__ == "__main__":
    main()
