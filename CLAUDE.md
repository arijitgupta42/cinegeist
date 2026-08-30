# CLAUDE.md

Working instructions for this repo. Read `plan.md` for the full design and rationale; this file is the operational summary.

## What this is

`cinegeist` — a conversational movie recommender for people who can't articulate their own taste. It learns preferences by getting the user to react to concrete films, stores a decaying taste profile locally, and searches a rich metadata catalog to find things to watch.

**The one-line design principle:** people are unreliable narrators of their own taste but excellent judges of specific items. Never ask abstract questions; ask people to react to concrete things.

## Hard rules

1. **The LLM never names a movie.** It receives candidate IDs from our retrieval layer and returns candidate IDs. Every title the user sees comes from the local catalog. Any ID returned by the model that isn't in the shortlist we sent is dropped, silently, with a log line. This is non-negotiable — hallucinated recommendations destroy the product.

2. **The maths decides, the LLM phrases.** Retrieval, scoring, question selection, and stopping rules are deterministic Python. The LLM does four things only: phrase a question, extract structured signals from free text, rerank a shortlist, write explanations. Don't move logic into prompts.

3. **Never hardcode an OpenRouter model ID as the default.** Free model slugs rotate constantly. Discover them at runtime from `https://openrouter.ai/api/v1/models`, filtering `pricing.prompt == "0"` and `pricing.completion == "0"`. The hardcoded list in `llm/registry.py` is an emergency fallback and is assumed stale.

4. **Budget one LLM call per conversational turn.** Free tiers are roughly 50 requests/day. If a feature needs a second call per turn, find another way.

5. **The API key never leaves the environment.** Read `OPENROUTER_API_KEY` from env. Never write it to a file, never log it, redact it in every exception path.

6. **Never commit** `.env`, `data/`, `*.db`, `*.npy`, or anything matching `sk-or-`.

7. **Profiles are append-only.** Write `preference_events` rows; never mutate a profile in place. The profile is a derived, cached view.

8. **The web demo has no LLM and no key.** It's a static GitHub Pages bundle, so anything in it is public. No API key, no OAuth flow, no proxy — questions are pre-phrased offline, answers are clicks, explanations are templated. If a web feature needs an LLM call, it doesn't go in the demo. Visitors who want free text install the CLI and bring their own key.

9. **Never pad a sparse region.** When the 2,000-film demo shard can't serve the user's taste, say so with numbers and show the nearest thing. Do not backfill with popular titles, do not relax the popularity penalty, do not add a minimum-results rule, and do not bias probe selection toward well-covered axes. A demo that always feels confident and a product that disappoints is the exact failure this avoids.

## Commands

```bash
make setup            # venv + install + pre-commit hook
make catalog          # build data/cinegeist.db and data/genome.npy (slow, resumable)
make catalog-refresh  # incremental TMDB update
make test             # unit tests, no network
make test-network     # integration tests, hits real APIs
make lint             # ruff check + format
make check            # lint + test, run before every PR
make web-shard        # regenerate the 2,000-film browser shard, then commit it

cinegeist chat             # the main experience
cinegeist chat --offline   # no LLM calls, fixed question phrasing
cinegeist models --free    # list currently-free OpenRouter models
cinegeist search "bleak cerebral 90s"   # debug catalog retrieval
cinegeist profile show     # print the taste profile with evidence
cinegeist profile forget <event-id>
cinegeist budget           # LLM calls used today
```

```bash
# in web/
npm run dev           # demo at localhost, needs a built shard
npm test              # vitest, including the shared spec/ fixtures
npm run build         # what the Pages workflow runs
```

## Layout

```
src/cinegeist/
  config.py           settings; precedence: --flag > env > config.toml > default
  llm/client.py       OpenRouter chat, retries, failover, redaction
  llm/registry.py     live free-model discovery, 24h cache
  llm/prompts/        one .md file per prompt — edit these, not inline strings
  catalog/build.py    pipeline orchestrator
  catalog/sources/    movielens.py, tmdb.py
  catalog/genome.py   numpy memmap, cosine, tag lookup
  profile/            store.py (events), model.py (dataclass), update.py (decay)
  convo/engine.py     state machine: seed → adaptive → constrain → present → feedback
  convo/probes.py     information-gain question selection
  convo/extract.py    free text → structured JSON signals
  recommend/          retrieve.py, score.py, rerank.py, explain.py

spec/                 shared fixtures — pytest AND vitest both read these
web/src/              static demo: score.ts, profile.ts, probes.ts,
                      coverage.ts, shard.ts, viz/{space3d,evidence,bars}.ts
scripts/build_web_shard.py
```

`score.ts`, `profile.ts`, and `probes.ts` are ports of their Python counterparts. **When you change one, change the other and update `spec/`** — the fixtures are what stops them drifting, and drift here is silent.

## Key concepts

**Tag genome.** MovieLens ships a dense ~1,100-dimension relevance matrix over ~13,000 films — tags like *atmospheric*, *cerebral*, *quirky*, *bleak*, *twist ending*. This is the taste space. A user profile is a weighted centroid in it; recommendation is cosine similarity. Stored as a float32 memmap at `data/genome.npy` (~59 MB). **Do not add FAISS** — a full scan is a single matmul in milliseconds.

**Two profile layers.** Long-term taste (persisted, 270-day half-life decay) and session intent (tonight's mood, time, company — discarded after the session). Session intent overrides long-term taste for the current recommendation but never writes to it. Someone who loves Tarkovsky still sometimes wants ninety minutes of nonsense.

**Information-gain probes.** For each candidate probe axis, compute how evenly it splits the high-scoring candidate pool, weighted by current profile uncertainty on that axis. Pick the max. Ground the question in two real films from the pool that sit at opposite poles. Then ask the LLM to phrase it.

**Stopping rules.** Stop when the top-5 is stable for two turns, or the rank-1-to-rank-10 margin clears the threshold, or 9 turns, or the user asks to stop. An escape hatch — `[just show me something]` — appears on **every turn** without exception. A returning user with a good profile gets picks in one turn.

**Output shape.** 3 confident picks + 1 wildcard. The wildcard is far from the profile centroid but shares 2+ strong tags; it prevents filter-bubble collapse and its accept rate tells us how adventurous the user is.

**Explanations must cite the user's own evidence.** "Same slow-burn dread you liked in *Hereditary*, but it resolves — you said ambiguous endings annoy you." Not "a gripping thriller." If an explanation could apply to any film, it's wrong.

**The browser demo works because the LLM is optional.** Scoring, probe selection, and stopping rules are deterministic, so the demo reimplements them in TypeScript against a 2,000-film shard and drops the LLM entirely. Questions are pre-phrased offline into `probes.json`, answers are clicks, explanations are templated from top contributing tags. It's a real recommender, not a mockup.

**Coverage honesty.** The shard build measures, per film, what fraction of that film's full-catalog neighbourhood survived into the shard, and packs it in as one byte. At recommendation time, if the weighted coverage near the profile centroid drops below 0.25 — or nothing in the shard is within 0.45 cosine — the demo takes the honesty path: state the numbers ("the demo catalog is 2,000 films; the full version searches about 13,000"), name the thin axes, relabel the header to "The closest the demo catalog gets", suppress the wildcard, and change nothing about the ranking. See hard rule 9. Probe selection never sees coverage data, so it can't steer visitors toward the dense region.

## Git workflow

- Only the initial scaffold commit goes directly to `main`. **Everything else: branch → PR → squash merge.**
- Branch names: `session-2/load-movielens-data`
- One PR = one coherent change. If the description needs "and" twice, split it.
- CI green before merge. Delete the branch after.

**Commit and PR messages are plain English sentences.** No `feat:`/`fix:`/`chore:` prefixes, no emoji, no jargon.

Good:
```
Load the MovieLens tag genome into a numpy file so searches are fast
Fall back to another free model when OpenRouter rate-limits us
Stop asking questions once the top five picks stop changing
```

Bad: `feat(catalog): implement ingestion pipeline` · `fix stuff` · `WIP` · `Update client.py`

PR body uses three headings: **What this changes** / **Why** / **How to test it**. The body explains *why*; the diff already shows *what*.

## Code style

- Python 3.11+, `ruff` for lint and format, type hints on all public functions.
- Dependencies: `httpx`, `pydantic`, `numpy`, `rich`, `typer`, stdlib `sqlite3`, `pytest`. **Adding anything else needs a reason in the PR description.**
- SQL lives in `.sql` files, not embedded strings. Prompts live in `.md` files, not inline.
- Scoring functions stay pure and deterministic — that's what makes them testable.
- Long-running jobs (catalog build, TMDB crawl) must be resumable after Ctrl-C and show progress.

## Testing

- Unit tests never touch the network; all HTTP is mocked. Integration tests are `@pytest.mark.network` and excluded from CI.
- `tests/fixtures/mini_catalog/` holds ~200 films with real genome vectors — small enough for git, real enough to test scoring against.
- Golden transcripts replay recorded conversations against a stubbed LLM so flow changes appear as diffs.
- Small models return malformed JSON regularly. Every LLM call that expects JSON: validate against a schema, retry once with the error appended, then fall back to deterministic behaviour. Never crash the conversation because a model misplaced a brace.
- `spec/` holds golden fixtures loaded by **both** `pytest` and `vitest`; they must agree within float tolerance and both run in CI. Changing a scoring rule means updating the fixtures, which forces both implementations to move together.

## Where the sessions are going

1. Skeleton, config, OpenRouter client with free-model discovery, CLI shell, CI
2. Catalog: schema, MovieLens ingest, genome memmap, TMDB enrichment, `cinegeist search`
3. Profile store with decay, seed questions, signal extraction, adaptive probes
4. Retrieval, scoring, MMR, rerank, explanations, feedback loop
5. Shared `spec/` fixtures, browser shard with coverage bytes, precomputed probes
6. The demo: Vite app, TypeScript scoring, click conversation, honesty path, Pages deploy
7. Visualization: bars, 3D taste map with barycenter marker, evidence graph, landing page
8. Buffer: eval harness with synthetic personas, predicted genome vectors, docs

Earlier sessions will overrun. Session 8 is the buffer; treat everything in it as optional.

## Attribution (required, don't remove)

TMDB attribution must appear in the README and CLI footer: *"This product uses the TMDB API but is not endorsed or certified by TMDB."* Cite Vig, Sen & Riedl (2012) for the tag genome and Harper & Konstan (2015) for MovieLens. Neither dataset is redistributed in the repo — we download at build time. See `DATA_LICENSES.md`.

## Privacy

Profile, history, and catalog stay local in SQLite with no telemetry. Prompts do go to OpenRouter and onward to model providers, and some free providers may log or train on inputs. Say this plainly in the README and keep `--offline` working.

The web demo sends nothing anywhere: no account, no server, no LLM calls, profile in `sessionStorage` and gone when the tab closes. Ship a "Clear everything" button and an "Export my session" button (which doubles as the migration path into the CLI). The only post-load network traffic is lazy-loaded posters from TMDB's CDN — mention that rather than claiming zero.
