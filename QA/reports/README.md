# QA Reports

Store generated QA outputs here.

Recommended pattern:

- Machine-readable JSON reports:
  - `YYYY-MM-DD_api_contract.json`
  - `YYYY-MM-DD_retrieval_baseline.json`
- Human-readable Markdown summaries:
  - `YYYY-MM-DD_manual_baseline.md`
  - `YYYY-MM-DD_api_contract.md`

The `QA/run_api_qa.py` script writes timestamped JSON and Markdown reports into this folder by default.
