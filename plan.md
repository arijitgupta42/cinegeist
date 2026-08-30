# CineGeist — Plan

A conversational movie recommender for people who know what they like but can't say what they like.

> **Name:** `cinegeist`. Package, CLI, and repo all use this — see `pyproject.toml` and the package directory.

---

## 1. The actual problem

The user's stated problem is "I can't decide what to watch." The real problem is that **stated preferences and revealed preferences diverge**. If you ask someone "what genres do you like?" they say "thrillers, I guess" and you hand them a list of thrillers they don't want. But if you show them *Prisoners* and *Now You See Me* and ask which they'd rather watch tonight, they answer instantly and correctly.

So the whole system is built on one rule:

> **Never ask abstract questions. Ask people to react to concrete things.**

People are unreliable narrators of their own taste but excellent judges of specific items. The conversation's job is to convert a handful of concrete reactions into a latent taste vector, and the recommender's job is to search a rich metadata space with that vector.

### Design consequences

| Instinct | What we do instead |
|---|---|
| "What genres do you like?" | "Pick the one you'd rather watch tonight: A or B" (with real films) |
| Ask 20 questions to build a full profile | Ask until the top-5 stops changing, then stop (usually 6–9 turns) |
| Let the LLM name movies | The LLM only ever picks from candidate IDs we retrieved. It never types a title from memory. |
| One profile per user | Two layers: **long-term taste** (slow, decaying) and **tonight's intent** (mood, time, company, throwaway) |
| Genre tags | The MovieLens tag genome: ~1,100 dimensions like *atmospheric*, *cerebral*, *quirky*, *bleak*, *visually appealing*, *twist ending* |

---

## 2. Data: what we build the catalog from

We need three things that no single source provides: **rich latent descriptors**, **current coverage**, and **a free licence**. So we join three sources.

### 2.1 MovieLens `ml-latest` — the taste dimensions (primary signal)

Download: `https://files.grouplens.org/datasets/movielens/ml-latest.zip`

The critical asset here is the **tag genome** (`genome-scores.csv` + `genome-tags.csv`). It's a dense matrix: every movie in the genome gets a relevance score in `[0,1]` for **every one of ~1,100 tags**. `ml-latest` ships roughly 14 million relevance scores across ~13,000 films. Tags include exactly the kind of thing users can't articulate — plot shapes (*character study*, *nonlinear*), texture (*gritty*, *stylized*, *slow*), affect (*melancholy*, *feel-good*, *disturbing*), craft (*great cinematography*, *good soundtrack*), and provenance (*based on a book*, *stanley kubrick*).

This is the single best public artifact for latent taste modelling. A user profile is a weighted centroid in this 1,100-dim space; recommendation is cosine similarity. Nothing else needed.

`links.csv` maps `movieId → imdbId → tmdbId`, which is our join key to everything else.

**Limitation:** the genome only covers well-rated, well-tagged films, and `ml-latest` regenerates periodically rather than daily. It will not contain last month's releases. Hence source 2.

### 2.2 TMDB — coverage, freshness, and structured facets

TMDB's API is free for non-commercial use (attribution required, key needed). It gives us, per film: genres, **keywords** (free-text, ~10–40 per film), full cast and crew, release date, runtime, original language, production countries and companies, collection membership, ratings, posters, and — usefully — regional **watch providers**.

TMDB also publishes daily ID export files and a `/changes` endpoint, so refreshing is incremental and cheap rather than a full re-crawl.

**Bridging the gap:** films too new or too obscure for the tag genome get a *predicted* genome vector. Fit a linear map from TMDB features (keywords + genres + decade + language, as a sparse one-hot) to genome vectors using the ~13k films where we have both, via ridge regression. Cheap, no GPU, works well enough. Predicted vectors are flagged `genome_source='predicted'` and get a small confidence discount at scoring time. **This is a Phase 2 item — do not build it in session 2.**

### 2.3 Wikidata — optional graph enrichment (Phase 2, only if there's time)

SPARQL endpoint, CC0, and genuinely graph-shaped: movements, awards, based-on relations, film-to-film influence. Nice for "more like this but weirder" traversals. Explicitly out of scope for sessions 1–4.

### 2.4 Storage

Two artifacts, both under `data/`, both gitignored:

- **`data/cinegeist.db`** — SQLite. Metadata, facets, credits, the user's profile and history. Roughly 300–600 MB depending on how much of TMDB you pull.
- **`data/genome.npy`** — a float32 numpy memmap, `n_movies × n_tags`. At 13k × 1,128 that's ~59 MB. A full-catalog cosine scan is a single matmul in a few milliseconds. **Do not install FAISS.** At this scale it's pure dependency cost for no gain.

### 2.5 Licences and attribution — do not skip

- TMDB requires the attribution line and logo. Put it in the README and in the CLI footer.
- MovieLens data is free for research/personal use; cite Vig, Sen & Riedl (2012) for the tag genome and Harper & Konstan (2015) for MovieLens.
- Ship a `DATA_LICENSES.md`. Neither dataset is redistributed in the repo — we download at build time.

---

## 3. Architecture

```
                  ┌──────────────────────────────────────────┐
   user  ◀────▶   │  CLI  (rich TUI)                         │
                  └────────────────┬─────────────────────────┘
                                   │
                  ┌────────────────▼─────────────────────────┐
                  │  Conversation engine (state machine)     │
                  │  seed → adaptive probes → constrain →    │
                  │  present → feedback                      │
                  └───┬────────────────────────┬─────────────┘
                      │                        │
        ┌─────────────▼──────────┐   ┌─────────▼──────────────┐
        │  Taste profile store   │   │  LLM layer (OpenRouter)│
        │  events → decay →      │   │  · phrase questions    │
        │  genome vector +facets │   │  · extract signals     │
        └─────────────┬──────────┘   │  · rerank + explain    │
                      │              └─────────┬──────────────┘
        ┌─────────────▼──────────────────────────────────────┐
        │  Recommender                                       │
        │  hard filters → genome cosine + facet boosts →     │
        │  MMR diversity → top ~40 → LLM rerank → 3 + 1      │
        └─────────────┬──────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────────────────────────────┐
        │  Catalog:  cinegeist.db (SQLite)  +  genome.npy    │
        └────────────────────────────────────────────────────┘
```

**The LLM is the mouth, not the brain.** Retrieval, scoring, and question choice are deterministic Python. The LLM phrases questions, parses free-text answers into structured signals, reranks a shortlist we handed it, and writes the explanations. This keeps it cheap (free models are rate-limited), reproducible, testable, and hallucination-free.

It also means the whole thing still works with the mouth removed, which is what makes the static browser demo (§8) possible: the demo reimplements the deterministic core in TypeScript, drops the LLM layer entirely, and is still a real recommender rather than a mockup.

### Repository layout

```
cinegeist/
├── CLAUDE.md
├── plan.md
├── README.md
├── LICENSE                      # MIT
├── DATA_LICENSES.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
├── data/                        # gitignored
├── src/cinegeist/
│   ├── cli.py                   # entry point
│   ├── config.py                # settings, env, ~/.config/cinegeist/config.toml
│   ├── llm/
│   │   ├── client.py            # OpenRouter chat, retries, failover
│   │   ├── registry.py          # live free-model discovery
│   │   └── prompts/*.md         # prompt templates, one file each
│   ├── catalog/
│   │   ├── schema.sql
│   │   ├── build.py             # pipeline orchestrator
│   │   ├── sources/{movielens,tmdb}.py
│   │   ├── genome.py            # memmap load, cosine, tag lookup
│   │   └── refresh.py           # incremental TMDB /changes
│   ├── profile/
│   │   ├── store.py             # events, snapshots
│   │   ├── model.py             # TasteProfile dataclass
│   │   └── update.py            # decay + recompute
│   ├── convo/
│   │   ├── engine.py            # state machine
│   │   ├── seed.py              # cold-start script
│   │   ├── probes.py            # information-gain question selection
│   │   └── extract.py           # free text → structured signals
│   ├── recommend/
│   │   ├── retrieve.py          # hard filters
│   │   ├── score.py             # cosine + facets + MMR
│   │   ├── rerank.py            # LLM rerank, ID-validated
│   │   └── explain.py
│   └── feedback.py
├── spec/                        # shared Python/TypeScript fixtures — see §8.6
├── web/                         # static browser demo — see §8.7
├── scripts/
├── tests/
│   └── fixtures/mini_catalog/   # ~200 films, committed, for offline tests
└── .github/workflows/ci.yml
```

**Stack:** Python 3.11+, `httpx`, `pydantic`, `numpy`, `rich`, `typer`, stdlib `sqlite3`, `pytest`, `ruff`. That's the whole dependency list and it should stay that short.

---

## 4. The taste model

### 4.1 Two layers, kept separate

**Long-term profile** (persisted, decays):
- `genome_vector`: 1,100-dim float32, the weighted centroid of liked films minus disliked films
- `facets`: director/writer/actor affinities, decade weights, country and language weights, runtime tolerance, subtitle tolerance, intensity ceilings (violence, gore, bleakness), pacing preference, ending preference (resolved vs ambiguous)
- `confidence`: per-axis, so we know what we still don't know

**Session intent** (thrown away after the session):
- available time, who's in the room, mood ("something that doesn't ask much of me"), platform constraints, "not in the mood for subtitles tonight"

A user who loves three-hour Tarkovsky films still sometimes wants ninety minutes of nonsense. Conflating these is the classic recommender failure. Session intent overrides long-term taste for tonight without ever writing to it.

### 4.2 Profile as an event log

Never mutate a profile in place. Append immutable evidence:

```sql
preference_events(
  id, user_id, ts, session_id,
  kind,      -- liked_movie | disliked_movie | pair_choice | axis_answer
             -- | constraint | post_watch_feedback
  subject,   -- movie id, tag id, or facet key
  value,     -- -1.0 .. +1.0
  weight,    -- initial confidence
  evidence   -- the user's own words, verbatim
)
```

The profile is a **derived view**, recomputed from the log:

```
w_i      = value_i × weight_i × 0.5^(age_days / HALF_LIFE)
vector   = Σ (w_i × genome_i) / Σ |w_i|
```

`HALF_LIFE = 270` days by default. This gives you drift for free — taste from two years ago fades without being deleted, and if the user comes back after a gap the old signal is quiet but recoverable.

Three things fall out of this design at no extra cost:
1. **Explainability.** "I'm suggesting this because of what you said about *Under the Skin* in March." The evidence string is right there.
2. **Correctability.** `cinegeist profile forget <event-id>` when the system latches onto something wrong.
3. **Auditability.** The user can read exactly what the system thinks it knows, in their own words.

Snapshots are cached in `profile_snapshots` and invalidated on new events, so we're not replaying the log on every turn.

---

## 5. The conversation

### 5.1 Shape: fixed spine, adaptive middle

Not fully scripted (can't adapt), not fully LLM-driven (rambles, burns rate limits, asks useless questions). Fixed opening, information-gain-driven middle, fixed close.

```
GREET
  └ new user      → SEED
  └ returning     → "Last time you were into X. Still true?" → SEED_LITE (1 probe) or straight to CONSTRAIN

SEED  (3 fixed questions, always the same)
  1. "Name two or three films you've genuinely loved." (free text, any era)
  2. "Anything you turned off halfway, or that everyone loves and you didn't?"
  3. A this-or-that pair, drawn from the catalog to straddle a major axis.

ADAPTIVE  (0–6 probes, chosen by information gain, early exit)

CONSTRAIN  (1 question: time available, who's watching, subtitles ok?)

PRESENT  (3 confident picks + 1 wildcard, each with a one-line reason)

FEEDBACK  ("watched and loved it" / "watched, it was fine" / "not for me" /
           "already seen it" / "show me three more")
```

**Question 2 is the most valuable question in the system.** Negative signal is sharper than positive signal — everyone says they love *The Godfather*, but "I bounced off *Blade Runner 2049* after forty minutes" tells you about pacing tolerance, and that's worth ten positive answers.

### 5.2 How adaptive probes get chosen

This is the interesting part, and it's math, not prompting.

1. Hard-filter the catalog into a candidate pool (typically 200–500 films) using current constraints.
2. Score the pool against the current profile vector.
3. For each of ~40 **probe axes** (high-variance genome tags in the pool, plus facets like decade and runtime), compute how evenly that axis splits the pool's *high-scoring* films. An axis where the top candidates are all identical tells us nothing; an axis that cleanly bisects them halves the search space.
4. Pick the axis with maximum expected entropy reduction, weighted by how uncertain the profile currently is on that axis.
5. **Ground the question in two real films** from the pool that sit at opposite poles of that axis and are otherwise similar.
6. Hand the axis and the two films to the LLM and ask it to phrase one natural, conversational question.

So the maths picks *what* to ask; the LLM picks *how* to ask it. The user experiences "how did it know to ask that?" and we get a question that actually reduces uncertainty.

**Guard:** if the user hasn't seen either film in a pair, that's still signal (obscurity tolerance) — offer a "neither, next" option and don't count it as a failure.

### 5.3 When to stop asking

Stop at the first of:
- the top-5 recommendation set is unchanged for two consecutive turns
- the score margin between rank 1 and rank 10 exceeds a threshold (we're confident)
- 9 total turns
- the user says anything resembling "just tell me already" — always honour this immediately

**Always show an escape hatch.** `[just show me something]` is available on every single turn. A recommender that holds your evening hostage behind a quiz is a worse product than one that guesses adequately in thirty seconds. A returning user with a good profile should be able to get picks in **one turn**.

### 5.4 Handling free-text answers

Free text goes to the LLM with a strict extraction prompt returning JSON only:

```json
{
  "mentioned_titles": [{"title": "...", "year": 2014, "sentiment": "loved"}],
  "axis_signals":     [{"axis": "slow", "value": -0.6, "quote": "..."}],
  "constraints":      [{"kind": "max_runtime", "value": 110}],
  "session_mood":     "wants something light"
}
```

Titles are then **resolved against the catalog by fuzzy match**, and ambiguous matches are confirmed with the user ("*Solaris* — Tarkovsky's or Soderbergh's?"). We never trust an LLM-emitted title as a catalog entry.

---

## 6. Recommendation pipeline

```
1. HARD FILTER      runtime, language, availability, release window,
                    exclude seen/rejected, content ceilings
                    → candidate pool

2. SCORE            0.55 × cosine(profile_vector, genome_vector)
                  + 0.20 × facet_match          (director, era, country, cast)
                  + 0.10 × quality_prior        (Bayesian-shrunk rating)
                  + 0.10 × session_fit          (tonight's mood/time)
                  − 0.05 × popularity           (discovery nudge)
                  × genome_confidence           (discount predicted vectors)

3. DIVERSIFY        MMR over genome vectors, λ=0.7
                    → prevents 3 picks that are the same film

4. SHORTLIST        top 40 → LLM rerank with profile evidence
                    LLM returns ordered IDs ONLY, validated against the shortlist

5. PRESENT          3 picks + 1 wildcard
                    wildcard = highest-scoring film that is *far* from the
                    profile centroid but shares 2+ strong tags
```

The wildcard matters. Pure centroid-matching converges on a filter bubble and gets boring by week two. The wildcard is a deliberate exploration slot, and its accept/reject rate is itself a useful signal (how adventurous is this user?).

### Explanations

One line per pick, and it must reference **the user's own evidence**, not generic marketing copy.

- Good: "Same slow-burn dread you liked in *Hereditary*, but it resolves — you said ambiguous endings annoy you."
- Bad: "A gripping thriller that will keep you on the edge of your seat!"

We pass the LLM the top contributing tags and the matching evidence quotes, and instruct it to reference at least one.

---

## 7. LLM layer (OpenRouter)

### 7.1 Free models are a moving target — never hardcode one

Free model IDs on OpenRouter rotate constantly. Providers that had free tiers six months ago have none now, and new ones appear. Any hardcoded `:free` slug will break.

**So: discover at runtime.**

```
GET https://openrouter.ai/api/v1/models
→ keep entries where pricing.prompt == "0" and pricing.completion == "0"
→ rank by a curated preference order (context length, instruction-following,
   JSON reliability), falling back to the live list order
→ cache to ~/.cache/cinegeist/models.json for 24h
```

Ship a small hardcoded fallback list for when the endpoint is unreachable, and treat it as a last resort that's probably stale.

### 7.2 Rate limits shape the architecture

Free tiers run around 50 requests/day (roughly 1,000 if you've bought credits), with per-minute throttles. That is a hard constraint, not a footnote. It's the main reason the scoring and question-selection logic is deterministic Python rather than agentic LLM calls.

**Budget: ≤ 1 LLM call per conversational turn.** A full session should cost 8–12 calls.

Mitigations:
- automatic failover to the next free model on 429 or 5xx
- exponential backoff with jitter
- prompt/response cache keyed on a hash of the inputs
- `--offline` mode: fixed question phrasing, no LLM at all, degraded but functional
- `cinegeist budget` shows calls used today

### 7.3 Model selection is the user's choice

```
Precedence:  --model flag  >  CINEGEIST_MODEL env  >  config.toml  >  auto-pick free
```

Any OpenRouter model ID works. Same code path for free and paid — the only difference is which ID goes in the request body. Config lives at `~/.config/cinegeist/config.toml`.

`OPENROUTER_API_KEY` is read from the environment. It is **never** written to a file, never logged, never committed. Redact it in all error output. It is also **never** bundled into the web demo, which has no LLM layer at all (§8.1).

### 7.4 Privacy note the README must carry

Prompts go to OpenRouter and onward to model providers, and some free providers reserve the right to log or train on inputs. Everything else — profile, history, catalog — stays local in SQLite with no telemetry. State this plainly and offer `--offline`.

---

## 8. The browser demo

A static, single-page version of CineGeist deployed to GitHub Pages, so someone can try the core interaction without installing anything.

### 8.1 The constraint: no server, no key

GitHub Pages serves static files. There is no backend, so there is nowhere safe to put an OpenRouter key — anything in the bundle is readable by anyone who opens devtools. Ship a key and it gets drained within a week.

**So no key ever ships, and the demo makes no LLM calls at all.** Not a degraded LLM path, not a rate-limited one — none. This isn't a limitation we're working around; it's the whole design, and §8.2 explains why it costs us almost nothing.

Two tiers were considered and deliberately rejected:

- **Sign in with OpenRouter (PKCE)** — visitors authorize, the page mints a key on *their* account, free text unlocks. Technically sound, but it puts a sign-in wall in front of a demo whose entire job is to be tried in ten seconds. Someone who wants free-text conversation can install the CLI and use their own key, which is a better outcome for them anyway.
- **A hosted proxy** (Cloudflare Worker + Turnstile + per-IP quota) — works, costs nothing, but it's a service we now operate and it destroys the "pure static site" property.

Both are recorded in §14 so they stay out.

### 8.2 What the demo is

The main architecture already does the heavy lifting here. **The LLM is the mouth, not the brain** — retrieval, scoring, probe selection, and stopping rules are all deterministic (§3). The LLM only phrases questions, extracts free text, reranks, and explains.

So a demo built on **pre-phrased this-or-that pairs between two real films** needs no LLM whatsoever:

| Job | In the CLI | In the demo |
|---|---|---|
| Phrase the question | LLM, per turn | precomputed offline into `probes.json` |
| Understand the answer | LLM extraction | it's a click — nothing to extract |
| Pick the next question | deterministic Python | deterministic TypeScript, same algorithm |
| Score and rank | deterministic Python | deterministic TypeScript, same algorithm |
| Explain a pick | LLM, cites evidence | templated from top contributing tags + the films that produced them |

That is a genuinely representative demo — it costs nothing to run, has no rate limit, works offline after load, and loads instantly. It also happens to showcase the best part of the product, since the pair mechanic *is* the core interaction. The parts it can't show are the phrasing and the prose, which are the least novel parts.

To show what it's missing, include **one recorded free-text transcript**, replayed on a timer, in a separate tab. Label it unmistakably as a recording. Do not fake it live.

### 8.3 The catalog shard

The full genome is 13,000 × 1,128 float32 ≈ 59 MB. Too big for a web page. `scripts/build_web_shard.py` reduces it:

```
1. Pick ~2,000 films: high genome confidence, spread across decades,
   countries and popularity bands. Deliberately include obscure ones —
   an all-blockbuster demo makes the recommender look stupid.
2. Truncated SVD of the genome matrix → 96 components.
3. Quantize to int8 with a per-component scale factor.
4. Keep top-12 tag IDs + scores per film, for explanations.
5. Compute per-film coverage (§8.4) against the full catalog.
6. Precompute UMAP 3D coordinates from the full genome vectors.
7. Emit shard.bin (packed int8) + shard.json (titles, years, runtimes,
   tmdb ids, poster paths, tag names, xyz coords, coverage bytes)
   + probes.json.
```

| Component | Size |
|---|---|
| SVD-96 int8 vectors, 2,000 films | 192 KB |
| Top-12 tags per film | 48 KB |
| UMAP xyz coordinates | 24 KB |
| Per-film coverage byte | 2 KB |
| Metadata (titles, years, ids) | ~150 KB |
| Precomputed probe pairs + phrasings | ~30 KB |
| **Total, gzipped** | **≈ 300 KB** |

Cosine similarity over 2,000 × 96 int8 is microseconds in plain JS. No WASM, no WebGPU, no library. Posters lazy-load from TMDB's CDN — don't bundle images.

Regenerate manually with `make web-shard` and commit the result. Don't run the catalog build in CI; it needs a TMDB key and takes far too long. Content-hash the shard filename so Pages can't serve a stale one.

### 8.4 Coverage honesty — saying so when the shard runs out

The demo catalog is 2,000 films. The real one is ~13,000. A visitor whose taste pulls toward Hungarian slow cinema, or 1970s exploitation, or anything else the shard sampled thinly, will land somewhere we cannot serve well. This is guaranteed to happen, not an edge case.

**The dishonest fix is to pad with popular titles.** Backfilling a sparse region with well-known films produces a demo that always feels confident and a product that disappoints on first real use. It also destroys the only signal that matters here — whether the pair mechanic actually locates people. So:

> **When the shard can't serve the user, the demo says so, in numbers, and shows the nearest thing it has.**

**Computing coverage (offline, at shard build time).** The build script holds the full genome, so it knows exactly what it is discarding:

```
for each film i kept in the shard:
  n_full(i)   = |{ j in full catalog : cosine(i, j) >= 0.85 }|
  n_shard(i)  = |{ j in shard        : cosine(i, j) >= 0.85 }|
  coverage(i) = n_shard(i) / n_full(i)        → quantize to one byte
```

One byte per film; 2 KB for the whole shard. This is a real measurement of what each neighbourhood lost, not a heuristic.

**Checking coverage (at recommendation time).** After scoring, before presenting:

```
region_coverage = Σ (w_k × coverage(k)) / Σ w_k
                  over the top-25 shard films nearest the profile centroid,
                  w_k = cosine(centroid, k)
```

Take the honesty path when **either**:

- `region_coverage < 0.25` — the shard kept under a quarter of this neighbourhood, or
- `top-1 cosine < 0.45` — the centroid has no close neighbour in the shard at all

Both thresholds live in `spec/` and are asserted by fixtures in both test suites, like every other constant.

**What the honesty path does.**

1. **States the numbers, above the picks.** *"The demo catalog is 2,000 films. Your taste is pointing somewhere it covers thinly — the full version searches about 13,000."*
2. **Names the direction, when it can.** The profile's top contributing tags are already computed for explanations; compare them against per-tag shard coverage and name the thin ones: *"You're pulling toward slow and eastern european — the demo shard has 14 films like that; the full catalog has around 300."*
3. **Shows the nearest thing anyway, labelled as such.** The header changes from "Your picks" to "The closest the demo catalog gets", and each reason says how far off it is instead of asserting a match.
4. **Suppresses the wildcard.** A deliberate exploration slot inside an already-sparse region is noise, not exploration.
5. **Does not backfill.** Ranking is unchanged, the popularity penalty is not relaxed, and there is no minimum-results rule promoting anything into the gap. If the honest answer is two mediocre matches, it shows two mediocre matches.

**And it does not steer.** Probe selection stays exactly as specified in §5.2 — maximum information gain over the pool. It would be easy, and very tempting, to bias probes toward axes the shard covers well, keeping visitors inside the dense region so the demo looks better. That is the same dishonesty as padding, moved one step earlier in the pipeline where it's harder to spot. **The probe layer does not get access to coverage data.**

**The map earns its keep here.** View A (§9.2) shades thinly-covered regions of taste-space, so a visitor walking into one *watches it happen* rather than being told afterwards.

This generalizes back to the CLI, where sparse regions also exist (films with predicted rather than measured genome vectors, §2.2). Same principle, same honesty: say what you don't know.

### 8.5 Browser storage

| What | Where | Lifetime | Size |
|---|---|---|---|
| Taste profile + event log | `sessionStorage` | tab close | ~10–30 KB |
| Turn counter | `sessionStorage` | tab close | trivial |
| Catalog shard | IndexedDB | until cleared | ~300 KB |
| Poster images | HTTP cache | browser-managed | lazy-loaded |

`sessionStorage` is per-tab and cleared on close — the definition of session-bound. 5 MB limit, strings only, so serialize the genome vector as base64.

The shard goes in IndexedDB instead because it's identical for every visitor, isn't personal data, and re-downloading it per tab is wasteful.

Ship a **"Clear everything"** button that wipes both stores, and an **"Export my session"** button that downloads the profile as JSON — which doubles as the migration path into the CLI.

### 8.6 Keeping two implementations honest

Scoring will now exist twice: Python (`src/cinegeist/recommend/score.py`) and TypeScript (`web/src/score.ts`). They will silently drift, and the failure mode is nasty — the demo quietly recommends different films than the real app and nobody notices for weeks.

Options considered: **Pyodide** (~10 MB download to run our own code — no); **a Rust core compiled to both a Python extension and WASM** (correct, elegant, roughly a whole extra session — note it as a Phase 3 option, don't do it now); **two implementations against a shared spec** — the right call at this budget.

```
spec/
  scoring/
    cases.json          # (profile, catalog, expected ranking) tuples
    decay.json          # (events, elapsed days, expected weights)
    probes.json         # (candidate pool, expected chosen axis)
    coverage.json       # (profile, shard coverage, expected honesty verdict)
  README.md             # the algorithm in prose, as the tiebreaker
```

Both `pytest` and `vitest` load the same JSON and assert identical output within float tolerance. Both run in CI. Changing scoring means updating the fixtures, which means both implementations move together or the build goes red.

**Create `spec/` before the TypeScript exists.** Writing fixtures while the Python is fresh is easy; reverse-engineering them later is archaeology.

### 8.7 Layout and deployment

```
cinegeist/
├── src/cinegeist/           # Python app (unchanged)
├── spec/                    # shared golden fixtures
├── web/
│   ├── index.html
│   ├── src/
│   │   ├── main.ts
│   │   ├── score.ts         # port of recommend/score.py
│   │   ├── profile.ts       # port of profile/update.py
│   │   ├── probes.ts        # port of convo/probes.py
│   │   ├── coverage.ts      # the honesty check (§8.4)
│   │   ├── shard.ts         # fetch, decode, IndexedDB cache
│   │   └── viz/
│   │       ├── space3d.ts   # three.js
│   │       ├── evidence.ts  # d3-force
│   │       └── bars.ts
│   ├── public/shard/        # generated, committed
│   ├── vite.config.ts
│   └── package.json
├── scripts/build_web_shard.py
└── .github/workflows/pages.yml
```

**Stack:** Vite + TypeScript, no framework — it's one page, React earns nothing here. `three` and `d3-force` are the only runtime dependencies. Target under 250 KB gzipped excluding the shard, with `three` lazy-loaded when the 3D tab opens. Note there is no `auth.ts` and no `llm.ts`; the demo has no network code beyond fetching the shard and lazy-loading posters.

**Deploy:** GitHub Actions on push to `main` — build, then `actions/deploy-pages`. Set Vite's `base` to `/cinegeist/` for a project page. Add `.nojekyll`.

---

## 9. Visualization

### 9.1 The idea

**Do not visualize the catalog. Visualize the user moving through it.** A static scatter of 2,000 films is wallpaper. A marker that visibly walks across taste-space as you answer each question is the demo.

### 9.2 The three views

**View A — Taste-space map (3D, the hero).** 2,000 films as points at their precomputed UMAP coordinates, coloured by dominant tag cluster. The user marker is the **barycenter** of the films they've reacted to — a weighted average of those films' 3D positions, using the same decayed weights as the real profile (§4.2). A fading trail shows the path taken across the conversation. Recommendations pulse; the wildcard pulses in a different colour, visibly further out. Thinly-covered regions (§8.4) are shaded, so walking into one is visible. Hover a point for title, year, top tags; click for "more like this."

*Why barycenter rather than projecting the profile vector?* UMAP is non-linear, so there's no clean way to project an arbitrary new point into it. The barycenter sidesteps that entirely, is one line of code, and is more honest anyway: *you are here, between these films you liked.* If you want a mathematically exact projection, use PCA instead — it's linear, so the profile vector projects with a matmul. PCA maps look worse but behave correctly. UMAP + barycenter is the right trade for a demo.

*Implementation:* three.js, one `THREE.Points` with a `BufferGeometry` and a custom shader for the glow. Two thousand points is nothing — this technique handles 100k. **Do not create a mesh per film.** `OrbitControls`, or hand-rolled drag-to-rotate to keep the bundle small.

**View B — Evidence graph (2D, the "why").** A node-link graph via `d3-force`:

```
[film you loved] ──┐
                   ├──▶ [tag: slow-burn] ──▶ ( YOU ) ──▶ [recommended film]
[film you loved] ──┘

[film you bounced off] ──▶ [tag: nonlinear] ──✗
```

Edge thickness is contribution weight; edge colour is positive or negative. Click a tag to highlight every film that contributed to it and show the verbatim quote from the event log. Faded, thinning edges show decay — old evidence literally looks faint.

This is `cinegeist profile show` rendered as a picture, and it's the single most trust-building thing in the product. People believe recommendations they can trace.

**View C — Axis bars (the boring one you'll actually use).** A ranked horizontal bar chart of the strongest tag affinities, positive and negative, with confidence shading. Ugly, legible, and the one people screenshot. Ten minutes of work. **Build it first.**

### 9.3 Shared rules

- **Same data source as the CLI.** The visualization reads the profile object, not a special view-model. If they can diverge, they will.
- Animate transitions between turns (~600 ms) so change is visible rather than a jump cut.
- 3D needs a 2D fallback for low-power devices and `prefers-reduced-motion`.
- Keyboard navigable; every visual state also reachable as text.

### 9.4 Landing page copy (draft)

> **CineGeist** — a movie recommender for people who can't explain their own taste.
>
> Instead of asking what genres you like, it shows you pairs of films and asks which you'd rather watch. After about eight clicks it knows more about you than a genre filter ever will — and it shows you exactly what it learned, and why.
>
> This demo runs entirely in your browser. No account, no server, nothing stored, no AI calls. Close the tab and it's gone. It searches a 2,000-film sample and will tell you when your taste points somewhere that sample covers thinly.
>
> The full version searches about 13,000 films, holds a conversation in plain language, and keeps a permanent profile that follows your taste as it changes. `pipx install cinegeist`
>
> *This product uses the TMDB API but is not endorsed or certified by TMDB. Movie tag data from MovieLens (GroupLens Research).*

---

## 10. Session-by-session build plan

Each session is a branch family. Every PR is small enough to review in one sitting. The listed "done when" is the acceptance test — don't move on without it.

### Session 1 — Skeleton and LLM plumbing

*The only session with a direct-to-main commit.*

- **Initial commit (direct to `main`):** README, LICENSE, .gitignore, plan.md, CLAUDE.md, pyproject.toml, empty package
- **PR — Add settings and config loading:** `config.py`, env + TOML precedence, `.env.example`
- **PR — Talk to OpenRouter:** `llm/client.py` with retries, timeouts, key redaction
- **PR — Find free models automatically:** `llm/registry.py`, live discovery, 24h cache, fallback list, failover on 429
- **PR — Add the command line shell:** `cinegeist models`, `cinegeist ask "..."`, `cinegeist config`
- **PR — Set up CI:** ruff + pytest on push and PR

**Done when:** `cinegeist models --free` lists live free models and `cinegeist ask "hi"` gets a reply, with no key in any log line.

### Session 2 — Build the catalog

- **PR — Create the database schema:** `schema.sql`, migrations helper
- **PR — Load MovieLens data:** download, verify, ingest movies + links + genome, write `genome.npy` memmap
- **PR — Enrich from TMDB:** keywords, credits, countries, providers; concurrent + rate-limited; resumable
- **PR — Add catalog search for debugging:** `cinegeist search "bleak, cerebral, 90s"` → tag-vector search over the catalog

**Done when:** `make catalog` builds `data/cinegeist.db` and `data/genome.npy` from nothing, is resumable after Ctrl-C, and `cinegeist search` returns results that make you nod.

### Session 3 — Profile and conversation

- **PR — Store what we learn about the user:** events table, snapshots, decay recompute, `cinegeist profile show|forget|reset`
- **PR — Add the opening questions:** `convo/seed.py`, title resolution with disambiguation
- **PR — Turn free text into signals:** `convo/extract.py`, strict-JSON prompt, schema validation, graceful failure
- **PR — Choose the next question by how much it teaches us:** `convo/probes.py`, information gain, pair grounding, stopping rules

**Done when:** a full conversation persists a profile, `cinegeist profile show` prints it with evidence quotes, and running twice produces a visibly sharper profile the second time.

### Session 4 — Recommend, explain, learn

- **PR — Find and rank candidate films:** `retrieve.py` + `score.py` + MMR
- **PR — Let the model pick the final order:** `rerank.py`, ID validation, drop anything not in the shortlist
- **PR — Explain each recommendation:** `explain.py`, must cite user evidence
- **PR — Learn from what the user did next:** `feedback.py`, post-watch capture, profile update
- **PR — Polish the conversation flow:** `rich` output, escape hatches everywhere, `--offline`

**Done when:** end-to-end `cinegeist chat` produces 3 + 1 picks with evidence-grounded reasons; giving negative feedback measurably shifts the next run.

### Session 5 — Shared spec and the browser shard

Sessions 1–4 above are unchanged by the web demo. This is where it starts.

- **PR — Write down the scoring rules as shared test data:** `spec/`, fixture generator, `pytest` reads them (§8.6)
- **PR — Build a small catalog for the browser:** `build_web_shard.py`, SVD, int8 quantization, UMAP coordinates
- **PR — Measure what the shard leaves out:** per-film coverage against the full catalog, packed into the shard (§8.4)
- **PR — Precompute the demo's questions:** probe pairs and phrasings generated offline into `probes.json`

**Done when:** `make web-shard` emits a shard under 400 KB gzipped carrying coverage bytes, and the Python scoring passes every `spec/` fixture.

### Session 6 — The demo itself

- **PR — Set up the web app and load the movie shard:** Vite scaffold, IndexedDB cache, `sessionStorage` profile
- **PR — Score movies in the browser:** port of scoring and decay, `vitest` against the same `spec/` fixtures
- **PR — Add the click-based conversation:** Tier 0 loop, probe selection, stopping rules, escape hatch
- **PR — Say so when the demo catalog runs thin:** the honesty path — banner, thin-axis naming, wildcard suppression, no backfill (§8.4)
- **PR — Publish the site to GitHub Pages:** Actions workflow, base path, `.nojekyll`, content-hashed shard

**Done when:** the deployed site runs a complete conversation and produces recommendations with **no LLM calls at any point** and no network traffic after load beyond lazy-loaded posters — and both test suites agree on scoring.

### Session 7 — Visualization

- **PR — Show taste as a bar chart:** View C, the quick win
- **PR — Show the map of taste-space:** three.js points, barycenter marker, trail, shaded thin regions, animated transitions
- **PR — Show why each recommendation was made:** `d3-force` evidence graph with quote-on-click
- **PR — Write the landing page:** what it is, the privacy statement, the 2,000-vs-13,000 caveat, TMDB attribution, link to the repo

**Done when:** a visitor can play through the demo, watch their marker move, click a tag and see why, and get told plainly when their taste lands somewhere the shard can't serve.

### Session 8 — Buffer, evaluation, docs

Assume the earlier sessions overrun. If they don't, spend it here.

- **PR — Add an offline test harness:** synthetic personas ("loves slow European drama, hates capes"), replayable transcripts, precision@3 tracking. This is how you tune the scoring weights without guessing.
- **PR — Predict tag vectors for films the genome doesn't cover:** ridge regression from TMDB features, confidence flag
- **PR — Write the docs and make it installable:** README with screenshots, `pipx install`

**Done when:** the eval harness runs on a fixture catalog in CI and reports a precision number you can watch move.

---

## 11. Development workflow

### Branch and PR rules

- **One direct commit to `main`, ever:** the initial scaffold. Everything after that is a branch → PR → merge.
- Branch names: `session-2/load-movielens-data`
- One PR = one coherent change. If the description needs the word "and" twice, split it.
- Squash merge. The PR title becomes the commit on `main`.
- Delete the branch after merge.
- CI must be green before merge.

### Commit and PR messages: plain English

Write the way you'd explain it to a colleague. **No `feat:` / `fix:` / `chore:` prefixes. No emoji. No jargon.**

Good commit messages:
```
Load the MovieLens tag genome into a numpy file so searches are fast
Fall back to another free model when OpenRouter rate-limits us
Stop asking questions once the top five picks stop changing
Fix the crash when a user names a film that isn't in the catalog
```

Bad:
```
feat(catalog): implement genome ingestion pipeline
fix stuff
WIP
Update client.py
```

The body answers *why*, not *what* — the diff already shows what.

PR descriptions use three headings: **What this changes**, **Why**, **How to test it**. Merge commits read as plain sentences: `Merge the catalog builder so the app has movie data to search`.

### Never commit

`.env`, `data/`, `*.db`, `*.npy`, `__pycache__/`, `.venv/`, `~/.config/cinegeist/`, any API key in any form. Add a pre-commit hook that greps for `sk-or-` and refuses.

---

## 12. Testing

- **Unit tests never touch the network.** All HTTP is mocked. Integration tests are marked `@pytest.mark.network` and excluded from CI by default.
- **Fixture catalog:** ~200 films with real genome vectors, committed to `tests/fixtures/`. Small enough for git, real enough to test scoring.
- **Golden transcripts:** recorded conversations replayed against a stubbed LLM, so conversation-flow changes show up as diffs.
- **The scoring function is pure and deterministic** given a profile and a catalog. Test it directly with hand-built profiles — this is where the real bugs live.
- **Shared spec fixtures** (`spec/`, §8.6) are loaded by both `pytest` and `vitest` and must produce identical output within float tolerance. Both jobs run in CI. This is the only thing stopping the Python and TypeScript scorers from drifting apart.

---

## 13. Known risks

| Risk | Mitigation |
|---|---|
| Free model IDs disappear mid-project | Runtime discovery + failover, built in session 1 (not bolted on later) |
| 50 requests/day kills iteration | `--offline` mode, response cache, deterministic core, fixture-based tests |
| Genome misses recent films | TMDB keywords as fallback signal; predicted vectors in Phase 2; be honest in the UI |
| Cold start is still cold | 3 seed questions + a decent quality prior beats nothing; the wildcard slot gathers signal fast |
| TMDB crawl is slow | Resumable, concurrent, rate-limited; ship a "popular 30k films only" default and a `--full` flag |
| Conversation feels like a survey | Hard turn cap, escape hatch on every turn, one-turn path for returning users |
| Filter bubble by week three | Wildcard slot, popularity penalty, MMR diversity |
| Small LLMs return malformed JSON | Schema validation, one retry with the error appended, then deterministic fallback |
| Someone extracts and abuses a shipped API key | No key is ever shipped. The demo makes no LLM calls at all — this is why Tier 0 is the whole demo (§8.1) |
| Python and TypeScript scorers drift apart | Shared `spec/` fixtures in both CI jobs — the entire reason `spec/` exists (§8.6) |
| 2,000-film shard can't serve a visitor's taste | Measured per-film coverage, an explicit "the demo catalog is 2,000 films" banner, and the nearest match shown as such. **Never padded with popular titles** (§8.4) |
| Demo shard skews popular and flatters itself | Films chosen for spread across decades, countries and popularity bands, not for recognisability |
| 3D visualization tanks on mobile | Lazy-load three.js, 2D fallback, honour `prefers-reduced-motion` |
| Pages serves a stale shard | Content-hash the shard filename at build time |

---

## 14. Out of scope (write it down so it stays out)

Accounts and multi-user auth. TV series. Group recommendations. Mobile apps. A trained neural recommender. Streaming-service deep links beyond TMDB's provider data. Social features. Anything that requires a server you have to pay for.

**Specifically rejected for the browser demo** (§8.1), so they don't get reopened every time someone notices the demo can't talk:

- **LLM access of any kind in the browser.** No shipped key, no "Sign in with OpenRouter" PKCE flow, no hosted proxy. Someone who wants free-text conversation installs the CLI and brings their own key. The sign-in wall costs more than the feature is worth in a ten-second demo.
- **A server of any kind**, including free-tier Workers. The demo is a static bundle or it isn't the demo.
- **Any server-enforced quota**, which follows from having no server. A client-side turn counter is a courtesy, not a control — it's three lines in devtools to reset — and that's fine, because Tier 0 has nothing worth abusing.

---

## 15. Success test

Two questions, asked after four weeks of real use:

1. Does a returning user get a genuinely good pick in **one turn**?
2. Can the system explain a recommendation in a way that makes the user say *"yes, that's exactly it"* — using something they actually said?

If both are yes, it works. Everything in this plan is downstream of those two sentences.

---

## References

- MovieLens datasets — https://grouplens.org/datasets/movielens/
- Tag Genome: Vig, Sen & Riedl (2012), *ACM TiiS* 2(3)
- MovieLens: Harper & Konstan (2015), *ACM TiiS* 5(4)
- TMDB API — https://developer.themoviedb.org/docs
- OpenRouter models endpoint — https://openrouter.ai/api/v1/models

*This product uses the TMDB API but is not endorsed or certified by TMDB.*
