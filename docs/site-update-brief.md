# Site Update Brief

## Goal

Make `disasterclippy.com` work better for a first-time visitor by:

- showing the product faster
- making trust feel concrete, not just claimed
- choosing a clearer primary audience on the marketing pages
- reducing repeated messaging
- guiding visitors into the right next step

## Recommended Positioning Shift

Current story:

`AI preparedness search with cited answers, reusable collections, and offline deployments`

Recommended story:

`A source-grounded knowledge system for preparedness and offline environments`

Why this is stronger:

- It makes the product feel more serious and durable than "chatbot."
- It centers the real differentiator: grounded answers with visible provenance.
- It creates room for both preparedness use and broader curated-knowledge use cases.
- It makes the chat interface feel like the access method, not the identity.

## Alignment With Existing Product Direction

This brief is meant to fit the direction already described in:

- `README.md`
- `docs/SUMMARY.md`
- `docs/deployment.md`
- `disaster-clippy-private/docs/design-direction.md`
- `disaster-clippy-private/docs/website-upgrades.md`

Important alignment points:

- The `.com` site is an orientation and inspiration surface, not the place for setup commands.
- The live demo is proof of concept for the product model, not the whole product story.
- The preparedness collection is the example deployment, not the only long-term use case.
- The site should progressively disclose the platform story instead of leading with repo or infrastructure details.
- The warm, library-like visual direction is a strength and should be preserved.

One adjustment from my initial critique:

- The homepage does not need to become consumer-first in a broad sense.
- It should become first-visit clear.
- That means product-first, trust-first, and proof-first, while still handing serious builders to docs and GitHub.

## Primary Audience Recommendation

The homepage should prioritize one visitor first:

`A curious evaluator asking: Is this trustworthy, useful, and worth trying?`

Secondary audiences should branch off after that:

- people who want to try the live demo
- people who want to self-host it
- people who want to use it with their own data

Right now the site speaks to all three too early, which makes the first impression less focused.

## Core Problems To Fix

### 1. Too much claim repetition, not enough product proof

The site repeats versions of:

- curated
- cited
- offline-capable
- bring your own data

Those are good messages, but they are stated more often than they are demonstrated.

### 2. The product appears as an idea before it appears as an experience

A first-time visitor should see one concrete example almost immediately:

- a real interface screenshot
- a real question
- a real answer excerpt
- a visible citation or source card

### 3. The information architecture is reasonable but not orchestrated

The navigation is fine, but the page-to-page journey is not intentional enough.

A first-time visitor should naturally move through:

1. what this is
2. why it is trustworthy
3. what sources are included
4. how to try it or build their own

### 4. Collections are explained structurally before they are made compelling

The current "collections/packs" story makes architectural sense, but a new visitor wants:

- what is in the demo
- why those sources were chosen
- what questions it can answer well
- what it does not cover

## What To Change First

### Priority 1: Rebuild the homepage around proof

Recommended homepage structure:

1. Hero
2. Product proof block
3. Why trust it
4. What is in the current collection
5. Who this is for
6. Pathways: try demo / explore sources / build your own

The biggest missing element is a proof block directly below the hero.

Example content for that block:

- a screenshot of the app with the answer and source panel visible
- a sample user question
- a short answer excerpt
- one highlighted source and why it was used

Headline direction:

`Ask a question. See the answer. Inspect the source.`

or

`Preparedness search that shows its work.`

This aligns especially well with the existing design-direction note that the product should feel like a good library:

- organized
- inspectable
- calm
- authoritative
- not flashy

### Priority 2: Make trust concrete

The site currently says the right things about trust, but it needs more receipts.

Add:

- a visible sample citation
- a short "how sourced answers work" sequence
- named source examples above the fold or near it
- a plain-language boundary statement

Example boundary statement:

`Disaster Clippy answers from the current collection only. If a source is not in the collection, it is not part of the answer.`

### Priority 3: Make the collections page earn its click

The collections page should feel like:

`Here is the library behind the demo`

not:

`Here is the conceptual packaging model`

Recommended additions:

- a named source list
- source categories
- inclusion criteria
- known boundaries and gaps
- sample questions the collection is good for

### Priority 4: Create stronger audience paths

Use explicit pathways after the core value proposition:

- `Try the demo`
- `Why trust the answers`
- `See the current sources`
- `Run it on your own data`

This gives each audience a clear next step without muddying the homepage narrative.

## Page-By-Page Recommendations

## Home

Keep:

- the overall tone
- the earth-toned visual identity
- the seriousness of the copy

Change:

- reduce repeated trust-language in the middle sections
- replace at least one value-card row with a proof section
- show a real screenshot instead of only a stylized showcase
- add a brief "who this is for" band
- make the CTA hierarchy clearer

Suggested hero framing:

Headline:

`A source-grounded knowledge system for preparedness.`

Subhead:

`Ask a question, get a usable answer, and inspect the source behind it. Built for curated collections, constrained environments, and offline use.`

Suggested CTA pair:

- `Try the Demo`
- `See the Sources`

Suggested support line:

`Not open-web AI. Not a black-box chatbot. Answers come from the collection in front of you.`

## About

Current strength:

- thoughtful explanation of the philosophy

Current weakness:

- too much conceptual restatement of ideas already present on the homepage

Recommended role for this page:

`Why this approach is trustworthy`

Suggested structure:

1. The problem with open-ended AI for preparedness
2. The design response: constrained collection plus visible sources
3. What Disaster Clippy does and does not do
4. Why hosted and local both matter
5. Who built it

This page should be a trust page first, not a philosophy essay.

## Collections

Current strength:

- introduces the idea that the searchable library is bounded and inspectable

Current weakness:

- still too abstract for a first-time visitor

Recommended role for this page:

`What is inside the demo and why it deserves trust`

Suggested additions:

- list the major named sources explicitly
- group them by type
- explain why each source family is useful
- show example question categories
- include a "not included" or "current limitations" section

This page can still mention packs, but that should be secondary.

## Docs

Current strength:

- clear as a directory

Current weakness:

- reads like a documentation index, not a guided decision point

Recommended improvement:

Split the top of the page into three entry paths:

- `Understand the product`
- `Run it locally`
- `Build your own collection`

This respects the current funnel:

`.com` -> proof and orientation -> technical docs -> GitHub

This would help both technical and non-technical visitors orient faster.

## FAQ

Current strength:

- answers good questions

Current weakness:

- repeats content found elsewhere

Recommended improvement:

- keep it concise
- remove long duplicate explanations
- use it to handle objections and edge questions

Examples:

- Is this a chatbot?
- What sources are included?
- Can it run offline?
- Can I use my own data?
- What are the limits of the current demo?

## Messaging Notes

### Phrases worth leaning into

- source-grounded
- shows its work
- bounded collection
- inspectable sources
- offline-capable
- curated knowledge

### Phrases to use more carefully

- chatbot
- AI assistant
- platform
- packs

These terms are not wrong, but they either feel generic or require too much explanation too early.

## Visual / UX Notes

The current visual language is good. It feels serious, warm, and distinct from generic AI sites.

Keep:

- serif-forward typography
- warm paper/earth palette
- restrained styling

Improve:

- add one or two real product screenshots
- use stronger visual proof blocks
- increase scanability with shorter paragraphs and subheads
- ensure every major section answers a different question

The update should not chase "more modern SaaS." The current visual tone is an asset.

## Suggested Information Architecture

Possible nav revision:

- Home
- Why Trust It
- Sources
- Docs
- Demo

If current labels remain, make their jobs sharper:

- `About` = trust model
- `Collections` = current sources and scope
- `Docs` = builder/deployer path

## Quick Wins

- add a real screenshot to the homepage
- add one sample cited answer on the homepage
- rename or reframe the collections page around sources
- add explicit source names near the top of the collections page
- tighten paragraph length across all public pages
- reduce duplicate "AI can be wrong" wording
- add a clearer "what the current demo covers" statement

## Easy Changes To Make First

These are the lowest-effort, highest-impact edits if the goal is to improve first impressions quickly without a full redesign:

### Copy-only

- change the homepage headline/subhead to emphasize source-grounded answers over generic AI language
- replace one repeated trust section with a short "how it works" sequence
- tighten paragraph lengths across `Home`, `About`, `Collections`, and `FAQ`
- move "not just a chatbot" higher on the homepage instead of leaving that idea mostly to FAQ copy
- add a plain-language scope statement for the current demo collection

### Content-only

- add one real product screenshot to the homepage
- add one real cited-answer example to the homepage
- add a named source list near the top of the collections page
- add a "what this collection is good for" block
- add a "current boundaries" block

### IA-only

- make `See the Sources` a more prominent CTA
- make the docs landing page branch by visitor intent
- ensure every page has one obvious next step instead of several equal-weight exits

## Metadata-Driven Opportunity

The private `website-upgrades.md` note points to a major opportunity that goes beyond copy polish:

- the system already has rich source and collection metadata
- the website currently exposes only a small editorial layer of that metadata

That suggests a strong next-stage improvement:

- make `Collections` a real, data-driven catalog
- surface trust and readiness fields publicly
- create collection profile pages backed by exported metadata snapshots

This would support the positioning extremely well because it turns "inspectable sources" into an actual product surface rather than just a promise.

## Proposed First Revision Sequence

1. Rewrite homepage structure and copy
2. Rework collections page into a source-trust page
3. Tighten about page around trust model and boundaries
4. Reframe docs landing page around visitor intent
5. Trim FAQ to reduce duplication

## Suggested Working Headline Set

Homepage options:

- `A source-grounded knowledge system for preparedness.`
- `Preparedness search that shows its work.`
- `Answers from a curated collection, with the source attached.`

Collections page options:

- `What the demo knows, and where it comes from.`
- `The sources behind the current collection.`

About page options:

- `Why this system is built around sources, not confidence.`
- `How Disaster Clippy keeps answers inspectable.`

## Bottom Line

The strongest future version of the site is not "an AI site that happens to mention citations."

It is:

`a serious, source-grounded knowledge product for preparedness and offline use`

The update should make that identity visible within the first screenful, then guide visitors into trust, scope, and action.
