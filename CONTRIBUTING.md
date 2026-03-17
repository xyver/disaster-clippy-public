# Contributing to Disaster Clippy

Disaster Clippy is an open preparedness search platform. Contributions of all kinds are welcome.

## Ways to contribute

**Use and test the platform**
- Use the hosted app at https://app.disasterclippy.com
- Try the local runtime on your own machine or a Raspberry Pi
- Report anything that does not work as expected via a GitHub issue

**Improve documentation**
- Fix errors, clarify confusing sections, or fill gaps in the docs
- Improve the README or any file in `docs/`
- The docs in `docs/` are the technical reference; the public site at disasterclippy.com is the product-facing version

**Suggest or review sources**
- Use the "Suggest a site" link in the hosted app to propose sources for new packs
- Review existing pack descriptions for accuracy or completeness

**Code contributions**
- Check open issues before starting significant work
- For small fixes (bugs, typos, broken behavior), open a PR directly
- For larger changes, open an issue first to discuss scope and direction
- The public repo contains the full runtime, local admin tooling, and source processing pipeline

## Project structure

```
disaster-clippy-public/
|-- app.py              main chat/search app
|-- admin/              local admin panel and source tools
|-- offline_tools/      indexing, embeddings, source management, translation
|-- cli/                command-line utilities
|-- docs/               technical documentation
|-- templates/          HTML templates
|-- static/             CSS and JS
|-- QA/                 test suite and evaluation rubrics
```

Read `docs/CONTEXT.md` for an architectural overview before making changes.

## Local setup

```bash
git clone https://github.com/xyver/disaster-clippy-public.git
cd disaster-clippy-public
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open http://localhost:8000 to verify the app runs.

## License

MIT for project code. Content sources keep their original licenses.
