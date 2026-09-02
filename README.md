# cinegeist

[![CI](https://github.com/arijitgupta42/cinegeist/actions/workflows/ci.yml/badge.svg)](https://github.com/arijitgupta42/cinegeist/actions/workflows/ci.yml)

Get a movie recommendation by reacting to real films instead of answering questions about
your taste. cinegeist shows you pairs of real films — *Prisoners* or *Now You See Me*? — learns
from your choices, and searches a rich metadata catalog for what to watch next.

It runs locally: the catalog and your taste profile live on disk in SQLite, and a free LLM is
used only to phrase questions and write explanations. The maths does the recommending, and the
model never invents a title — every film you see comes from the local catalog.

**Status:** active development. The CLI (catalog, profile, conversation, recommendations) and the
browser demo are in place.

- Try it with no install and no key: the [browser demo](#browser-demo) in [`web/`](web/).
- Prefer the terminal: [`cinegeist chat`](#quickstart). It works offline too (`--offline`).

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Configuration](#configuration)
- [Browser demo](#browser-demo)
- [Development](#development)
- [Privacy](#privacy)
- [Attribution](#attribution)
- [Licence](#licence)

## Requirements

- Python 3.11 or newer.
- ~1 GB of disk for the catalog (MovieLens data, a genome memmap, and the SQLite database).
- Optional: an [OpenRouter](https://openrouter.ai/keys) API key (free) so chat can phrase questions
  and explanations. Without one, chat runs offline with fixed phrasing.
- Optional: a [TMDB](https://www.themoviedb.org/settings/api) API key to enrich the catalog with
  keywords, credits, countries, and watch providers.

## Install

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Keys are read from the environment (both optional — see [Configuration](#configuration)):

```bash
export OPENROUTER_API_KEY=...   # Windows PowerShell: $env:OPENROUTER_API_KEY = "..."
export TMDB_API_KEY=...
```

Copy [`.env.example`](.env.example) to `.env` for a template. Keys are read from the environment
only — never written to a file, never logged, and redacted in error output.

## Quickstart

```bash
# 1. Build the local catalog. Downloads MovieLens; the first run is slow and resumable.
#    TMDB enrichment runs at the end if TMDB_API_KEY is set (otherwise it's skipped).
make catalog                    # same as: cinegeist catalog build

# 2. React to a few films and get tonight's picks: 3 confident + 1 wildcard, each with a reason.
cinegeist chat                  # uses OPENROUTER_API_KEY; without a key it runs offline

# No key, or want no LLM calls at all:
cinegeist chat --offline
```

See what it learned about you, in your own words:

```bash
cinegeist profile show
```

## Commands

Run `cinegeist --help`, or `cinegeist COMMAND --help`, for the full signature. Everything runs
locally. `--data-dir PATH` points any command at a catalog outside the default `./data`.

### Recommend and converse

| Command | What it does |
|---|---|
| `cinegeist chat` | The main experience: react to film pairs, get 3 picks + 1 wildcard with reasons. |
| `cinegeist chat --offline` | The same recommender with no LLM calls and fixed phrasing; works without a key. |
| `cinegeist chat -m <model>` | Pin a specific OpenRouter model instead of auto-selecting a free one. |

### Inspect and correct your taste

| Command | What it does |
|---|---|
| `cinegeist profile show` | Print your strongest affinities and aversions, with your own words as evidence. |
| `cinegeist profile show -n 12` | Show more axes per side (default 8). |
| `cinegeist profile forget <event-id>` | Delete one piece of evidence; ids come from `profile show`. |
| `cinegeist profile reset` | Erase the profile and start over. Asks first; `--yes` skips the prompt. |

### Build the catalog

| Command | What it does |
|---|---|
| `cinegeist catalog build` | Download MovieLens and build `data/cinegeist.db` and `data/genome.npy`. Resumable. |
| `cinegeist catalog build --skip-enrich` | Build the genome only; skip the TMDB stage. |
| `cinegeist catalog build --skip-predict` | Skip predicting genome vectors for films the tag genome doesn't cover but TMDB does. |
| `cinegeist catalog enrich` | Fetch keywords, credits, countries, and providers from TMDB. Concurrent and resumable. |
| `cinegeist catalog enrich --scope all` | Enrich every film, not just the genome-covered ones. |

### Debug and inspect

| Command | What it does |
|---|---|
| `cinegeist search "bleak cerebral 90s"` | Rank catalog films by tag-genome similarity to your phrase. No LLM, no profile. |
| `cinegeist eval` | Score the recommender against synthetic personas and print precision@3. Self-contained — no catalog or key. `--seed` for a different fixture, `--verbose` for the transcripts. |
| `cinegeist models --free` | List OpenRouter models that are free right now, best first (`--refresh` to skip the cache). |
| `cinegeist ask "hi"` | Send one message to a free model and print the reply (needs a key). |
| `cinegeist config` | Show the effective settings. Keys appear only as set / not set, never printed. |

## Configuration

Settings resolve in this order, first match wins:

```
--flag  >  environment variable  >  ~/.config/cinegeist/config.toml  >  built-in default
```

| Setting | Environment variable | Default | Used for |
|---|---|---|---|
| Model | `CINEGEIST_MODEL` | auto (a free one) | Any OpenRouter model id. `--model` / `-m` overrides it. |
| OpenRouter key | `OPENROUTER_API_KEY` | — | Phrasing and explanations in `chat`, and `ask`. |
| TMDB key | `TMDB_API_KEY` | — | `catalog enrich` (or `TMDB_ACCESS_TOKEN` for a bearer token). |
| Catalog location | `CINEGEIST_DATA_DIR` | `./data` | Where `cinegeist.db` and `genome.npy` live. `--data-dir` overrides. |

`cinegeist config` prints the resolved values and the paths to the config file and data directory.

## Browser demo

A static, no-install version of the core interaction lives in [`web/`](web/). It reimplements the
deterministic recommender in TypeScript against a 2,000-film sample and drops the LLM entirely —
questions are pre-phrased, answers are clicks — then adds three visualizations of your taste: a
diverging bar chart, a 3D taste-space map, and an evidence graph tracing each pick back to what you
chose. No account, no server, no AI calls; the session lives in the browser tab and is gone when you
close it.

```bash
cd web
npm install
npm run dev                     # open the printed localhost URL
npm test                        # vitest; shares the spec/ fixtures with the Python suite
npm run build                   # the static bundle the Pages workflow ships
```

The demo needs a built shard in `web/public/shard`; one is committed, and `make web-shard`
regenerates it from a local catalog.

## Development

The task runner is a `Makefile`. Without `make` (common on Windows), run the direct command instead.

| Task | `make` | Direct command |
|---|---|---|
| venv + install + git hook | `make setup` | see [Install](#install), then `git config core.hooksPath .githooks` |
| Lint and format check | `make lint` | `ruff check src tests && ruff format --check src tests` |
| Auto-format | `make format` | `ruff format src tests && ruff check --fix src tests` |
| Unit tests (no network) | `make test` | `pytest` |
| Integration tests | `make test-network` | `pytest -m network` |
| Everything CI runs | `make check` | `ruff check src tests && ruff format --check src tests && pytest` |
| Regenerate shared fixtures | `make spec` | `python scripts/build_spec_fixtures.py` |
| Rebuild the browser shard | `make web-shard` | `python scripts/build_web_shard.py` |
| Demo tests / build / dev | `make web-test` / `web-build` / `web-dev` | `cd web && npm test` / `npm run build` / `npm run dev` |

Unit tests never touch the network; all HTTP is mocked. The Python scorer and its TypeScript port
are held together by shared golden fixtures in [`spec/`](spec/), which both `pytest` and `vitest`
assert in CI — change a scoring rule without regenerating them and one job goes red.

## Privacy

Your profile, history, and catalog stay local in SQLite with no telemetry. The one thing that leaves
your machine, when online, is the text of your prompts: they go to OpenRouter and on to model
providers, and some free providers may log or train on inputs. Run `cinegeist chat --offline` to make
no LLM calls at all.

The browser demo sends nothing anywhere — no account, no server, no LLM. The only network traffic
after load is posters, lazy-loaded from TMDB's CDN.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB. Movie tag data is from
MovieLens (GroupLens Research). See [`DATA_LICENSES.md`](DATA_LICENSES.md) for citations and terms.
Neither dataset is redistributed here; both are downloaded at build time.

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code) by [Anthropic](https://anthropic.com).

## Licence

MIT — see [`LICENSE`](LICENSE).
