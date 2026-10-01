# U.S. Figure Skating rulebook extraction

The included [sources.yaml](sources.yaml) describes the 451-page **2026-27 U.S. Figure Skating Rulebook**. It is a reproducible starting point for a cited rules search, based on the PDF supplied for this project. The rulebook PDF and generated SQLite indexes stay outside version control.

## Source

- Publisher page: https://usfigureskating.org/sports/2025/8/9/rulebook-bylaws.aspx
- Publisher's direct PDF: https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/usafs.sidearmsports.com/documents/2026/8/13/2026-27_Rulebook.pdf
- Tested local file: `E:\Downloads\2026-27_Rulebook.pdf`
- Tested file size: 9,947,564 bytes; SHA256: `b701a4f83527a394b4ef5486198dc52c2feb11232a9d7b463e83602d388c83f3`

The publisher says its online rulebook can receive corrections after publication. It also says several element requirements are maintained in separate charts rather than in this PDF. Treat this index as an edition-specific snapshot and add those supplemental documents as separate sources before claiming complete rules coverage.

## Layout found in this PDF

| PDF pages | Content | Extraction |
| --- | --- | --- |
| 1-2, 13-14 | Introductory material | Separate page chunks |
| 15-242 | Bylaws and sport rules | Rule/section chunks, including `GR 1.00`, `PSER 1.01`, and `9042` |
| 245-328 | Skating skills diagrams | Individual-page chunks |
| 330-451 | Pattern dance diagrams | Individual-page chunks |

Table-of-contents pages 3-12, 243-244, and 329 are skipped. Image-heavy diagrams have incomplete searchable text; their PDF pages remain linked for visual inspection.

## Run with your PDF

Copy the PDF to `examples/skating/pdfs/2026-27_Rulebook.pdf` or copy `sources.yaml` to an ignored `sources.local.yaml` beside it and replace each `path` with your absolute PDF path. The manifest uses local `all-MiniLM-L6-v2` embeddings. Install the `local` extra and download the model once before building; subsequent builds and queries can run offline. The fictional sample uses hash embeddings for a dependency-light demonstration.

From the `clippy-core` directory:

```powershell
python -m pip install -e '.[pdf,local]'
python -m clippy_core.cli preview -m examples/skating/sources.yaml --doc usfs-bylaws-and-rules --limit 30
$env:CLIPPY_MODEL_CACHE = (Join-Path (Get-Location) 'build/models')
python -m clippy_core.cli build -m examples/skating/sources.yaml --rebuild
python -m clippy_core.cli eval examples/skating/golden.yaml -m examples/skating/sources.yaml --compare
python -m clippy_core.cli search 'senior synchronized short program time' -m examples/skating/sources.yaml
```

Once the model is downloaded, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent network checks during local builds and queries. Set `CLIPPY_MODEL_CACHE` each time so queries load the same model. The index records its embedder and refuses a mismatched one.

The starter [golden.yaml](golden.yaml) has eight questions with rule IDs checked against this PDF. It tests retrieval, not LLM answer quality. Expand it with real skater, coach, and official questions before relying on generated answers.
