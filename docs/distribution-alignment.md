# Distribution Alignment

This note captures what the recent DaedalMap distribution-model updates clarify, and
which of those lessons apply directly to Disaster Clippy.

It is not a replacement for [`deployment.md`](deployment.md) or
[`source-pack-release-policy.md`](source-pack-release-policy.md). It is the bridge
between the shared distribution-model work and this public repo's current state.

Terminology used here:

- `source pack`: the distributable content unit
- `catalog`: the remotely published list of source packs
- `installed`: present in the local data root
- `active`: enabled for the current runtime experience

---

## What Changed In DaedalMap

The recent DaedalMap updates sharpened the distribution model in a few important ways:

- the wrapper now targets a controlled engine release artifact rather than "whatever is
  on GitHub"
- engine install, integrity verification, local launch, and relaunch are treated as one
  explicit loop
- the first user-facing pack-store loop now exists end to end
- pack state is split clearly into:
  - available/store catalog
  - installed/local library
  - active/runtime catalog
- optional heavy capabilities stay layered behind the base app and pack install flow

That bundle moved from "target architecture" toward "working loop plus next milestone."

---

## What Already Maps Cleanly To Clippy

Several of those ideas already fit the current public repo well:

### 1. One Engine, Multiple Delivery Paths

Clippy already behaves like one maintained engine with multiple runtime modes rather
than separate apps:

- hosted/public runtime
- local runtime
- global maintainer runtime

That is the same architectural guardrail reinforced in the DaedalMap updates.

### 2. Pack Store As A Real Product Surface

Clippy already has meaningful pack-distribution building blocks:

- public catalog consumption in [`app.py`](../app.py)
- source-pack browsing and install UI in [`admin/templates/packs.html`](../admin/templates/packs.html)
- pack download/install routes in [`admin/routes/packs.py`](../admin/routes/packs.py)
- existing source-pack release policy in
  [`source-pack-release-policy.md`](source-pack-release-policy.md)

This means the "real storefront, not just loose downloads" lesson applies immediately.

### 3. Layered Optional Capabilities

Clippy already separates base runtime from heavier optional layers:

- source tools
- local models
- language packs
- translation and video processing workflows

That matches the shared layered-install model and does not need a conceptual rewrite.

### 4. Clear Writable Data Ownership

Clippy's `BACKUP_PATH` model is already a strong answer to a question the wrapper will
need to solve:

- where packs live
- where models live
- where translated/cache artifacts live
- how users move data to external or offline media

This is one area where Clippy is already better-positioned than many greenfield wrapper
plans.

---

## The Main DaedalMap Lessons Clippy Should Adopt

### 1. Make The Engine Release Loop Explicit

DaedalMap's biggest practical improvement is not just "have a wrapper." It is having a
clear loop:

```text
controlled manifest
  ->
download engine artifact
  ->
verify hash/signature
  ->
install locally
  ->
launch and relaunch
  ->
offer update when manifest changes
```

For Clippy, this should become the default local-install story.

That aligns with the existing rules already documented in
[`deployment.md`](deployment.md):

- controlled manifests
- signed artifacts
- explicit promotion before release visibility

The DaedalMap update is useful because it turns those from abstract principles into a
concrete product loop Clippy should mirror.

### 2. Treat Source-Pack State As Three Different Things

Clippy should adopt the same state split DaedalMap now uses:

- available/store catalog
- installed/local library
- active/runtime catalog

The current repo already has pieces of this, but the model should be made explicit in
future wrapper and admin work.

Why this matters:

- browsing what exists in the catalog is different from knowing what is on disk locally
- having a source pack downloaded is different from having it enabled in the runtime
- update checks and uninstall behavior get much simpler once these states are not mixed

### 3. Define A "First Working Loop" Milestone

DaedalMap improved by stating the first real proof point very clearly.

For Clippy, the equivalent milestone should be:

**wrapper installs a signed engine artifact, launches successfully, and lets the user
add at least one verified pack without manual filesystem work**

That milestone is better than a vague "build the wrapper" goal because it proves:

- packaging
- trust and verification
- local launch
- pack acquisition
- real user value after install

### 4. Keep Optional Heavy Downloads Behind The Base App

DaedalMap now frames local AI as an optional layer after the app and packs are already
useful.

Clippy should keep doing the same with:

- source tools
- local LLMs
- language packs
- larger translation resources

The base app plus packs should be the first-run success condition. Heavy extras should
remain opt-in.

### 5. Build For Air-Gapped Artifact Transfer, Not Just Online Download

Clippy's offline and Raspberry Pi story is stronger than DaedalMap's, so this lesson
needs to be applied even more aggressively here.

The wrapper and catalog model should assume both paths are first-class:

```text
online manifest/download/install
```

and

```text
download elsewhere
  ->
move by USB or local folder
  ->
install from local artifact
```

That should apply to:

- engine releases
- source packs
- local models
- language packs where feasible

---

## What This Repo Already Suggests We Should Build Next

Based on the current public repo, the highest-leverage distribution follow-ups are:

1. Define a stable engine release manifest contract for Clippy's downloadable runtime.
2. Add a signed-artifact installer/launcher path around the existing local runtime.
3. Normalize pack/catalog state around available, installed, and active concepts.
4. Keep model and language downloads in the same storefront mental model, but not in the
   minimum install.
5. Preserve `BACKUP_PATH` as the canonical local data root for packs, models, and caches.

---

## Practical Interpretation

The DaedalMap changes do not suggest that Clippy needs a different architecture.

They suggest that Clippy should take its existing strengths:

- modular runtime modes
- pack install surfaces
- `BACKUP_PATH` data ownership
- optional model/language layers
- release-policy thinking

and turn them into a more explicit first working distribution loop with stronger
artifact, manifest, and pack-state vocabulary.

That is the most transferable part of the recent DaedalMap work.
