# Template Deployments

We are working on making Disaster Clippy easier to fork and deploy as a starting point for your own local knowledge system.

## What this means

Right now, getting a local instance running requires setting up the full runtime, configuring environment variables, and understanding the source pack model before you can do much with it. That is fine for developers who want to work with the whole system, but it is a higher barrier than it needs to be for someone who just wants to run their own offline knowledge base with a custom set of sources.

The goal is a stripped-down deployment path that:
- starts with one pre-loaded source pack so the app is useful immediately
- requires no admin setup to get a working search experience
- is clearly structured as a base to build from, not just a hosted app to run

## What we are building toward

The foundation for this is the clippy-core extraction work documented in [clippy-core-extraction.md](clippy-core-extraction.md). The idea is to separate the portable runtime core from the advanced local admin and source-building tooling, so a developer can take just the parts they need.

Once that seam is clean, a template deployment would look something like:

1. Fork the template repo
2. Drop in a source pack
3. Run with a single command
4. Extend or customize from there

## Current state

The existing quick start in the README gets you running locally today. If you want to experiment with self-hosting now, that is the right starting point. The template path is a refinement of that, not a prerequisite for it.

If you are building something on top of this and have feedback on what would make that easier, open an issue or start a thread in Discussions.
