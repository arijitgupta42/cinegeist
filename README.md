# cinegeist

[![CI](https://github.com/arijitgupta42/cinegeist/actions/workflows/ci.yml/badge.svg)](https://github.com/arijitgupta42/cinegeist/actions/workflows/ci.yml)

A conversational movie recommender for people who know what they like but can't say
what they like.

Ask someone "what genres do you like?" and they say "thrillers, I guess" and you hand
them a list they don't want. Show them *Prisoners* and *Now You See Me* and ask which
they'd rather watch tonight, and they answer instantly and correctly. cinegeist is
built on that difference:

> **Never ask abstract questions. Ask people to react to concrete things.**

It learns your taste from a handful of concrete reactions, stores a decaying taste
profile locally, and searches a rich metadata catalog (the MovieLens tag genome, plus
TMDB) to find things to watch. A language model phrases the questions and writes the
explanations — but the maths decides what to recommend, and **the model never names a
film from memory**; every title you see comes from the local catalog.

See [`plan.md`](plan.md) for the full design and rationale, and
[`CLAUDE.md`](CLAUDE.md) for the working rules.

## Status

Early development. **Session 1** is in place: settings and config loading, the
OpenRouter client with retries and key redaction, automatic free-model discovery, the
command-line shell, and CI. The catalog, profile, conversation, and recommender arrive
in later sessions.

Working today:

```bash
cinegeist models --free    # list currently-free OpenRouter models
cinegeist ask "hi"         # one-shot chat through a free model
cinegeist config           # show effective settings (key redacted)
```

## Install

Requires Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Set your OpenRouter API key in the environment (needed for `ask`, not for
`models --free`):

```bash
export OPENROUTER_API_KEY=...   # Windows PowerShell: $env:OPENROUTER_API_KEY = "..."
```

Copy [`.env.example`](.env.example) to `.env` for a template. The key is read from the
environment only — cinegeist never writes it to a file, never logs it, and redacts it
in error output.

## Developing

The task runner is a `Makefile`. If you don't have `make` (common on Windows), run the
underlying commands directly:

| Task | `make` | Direct command |
|---|---|---|
| Set up venv + install + git hook | `make setup` | see **Install** above, then `git config core.hooksPath .githooks` |
| Lint and format check | `make lint` | `ruff check src tests && ruff format --check src tests` |
| Auto-format | `make format` | `ruff format src tests && ruff check --fix src tests` |
| Run unit tests (no network) | `make test` | `pytest` |
| Run integration tests | `make test-network` | `pytest -m network` |
| Everything CI runs | `make check` | `ruff check src tests && ruff format --check src tests && pytest` |

Unit tests never touch the network; all HTTP is mocked. Integration tests are marked
`network` and are excluded from the default run and from CI.

## Privacy

Your profile, history, and catalog stay local in SQLite with no telemetry. The one
thing that leaves your machine is the text of your prompts: they go to OpenRouter and
onward to model providers, and some free providers may log or train on inputs. Run with
`--offline` (arriving in a later session) to make no LLM calls at all.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB. Movie tag data
is from MovieLens (GroupLens Research). See [`DATA_LICENSES.md`](DATA_LICENSES.md) for
citations and terms. Neither dataset is redistributed here; both are downloaded at build
time.

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code) by [Anthropic](https://anthropic.com)

## Licence

MIT — see [`LICENSE`](LICENSE).
