"""Enrich the catalog from TMDB.

MovieLens gives us the tag genome; TMDB fills in the facets the recommender filters and
explains with — keywords, cast and crew, production countries, streaming providers, plus the
core details (runtime, language, ratings, collection). One HTTP call per film does it, using
``append_to_response`` to pull details, keywords, credits, and watch providers together, so
the whole job stays within one request per movie.

Two rules shape the code here:

* **Concurrent but polite.** Films are fetched through a bounded thread pool behind a shared
  rate limiter. Only the fetch (HTTP + JSON) happens on the workers; every database write
  goes through the single calling thread, so the one SQLite connection is never touched
  concurrently.
* **Resumable.** Each film's ``tmdb_fetched_at`` is stamped when it is written and progress
  is committed in batches, so a Ctrl-C (or a crash) loses at most the last uncommitted batch
  and the next run picks up exactly where it left off.

The TMDB credential is read from the environment only and redacted out of every error, just
like the LLM key.
"""

from __future__ import annotations

import random
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ...config import Settings

DEFAULT_API_BASE = "https://api.themoviedb.org/3"
# Everything we need for one film in a single request.
_APPEND = "keywords,credits,watch/providers"

# Retryable transient statuses (rate limiting and transient server errors).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_BASE_BACKOFF_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 8.0

# Credits are trimmed to what the taste model uses: the top-billed cast and the crew jobs that
# drive director/writer/etc. affinities. Storing every grip would bloat the catalog for nothing.
_MAX_CAST = 15
_KEEP_CREW_JOBS = frozenset(
    {
        "Director",
        "Writer",
        "Screenplay",
        "Story",
        "Author",
        "Novel",
        "Director of Photography",
        "Original Music Composer",
        "Editor",
        "Producer",
    }
)


class TMDBError(RuntimeError):
    """Base class for TMDB client failures. Messages are always credential-redacted."""


class TMDBAuthError(TMDBError):
    """The TMDB credential is missing or was rejected (401/403). Not worth retrying."""


class _RateLimiter:
    """A tiny thread-safe minimum-interval limiter shared across the worker pool."""

    def __init__(self, per_second: float | None) -> None:
        self._interval = 0.0 if not per_second else 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, *, sleep: Callable[[float], None], monotonic: Callable[[], float]) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        wait = start - now
        if wait > 0:
            sleep(wait)


class TMDBClient:
    """Fetches one enriched movie record from TMDB, with retries and a shared rate limit.

    Inject ``transport`` (an ``httpx.MockTransport``) in tests, and ``sleep``/``monotonic``/
    ``rng`` to make retries and rate limiting deterministic.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        api_base: str = DEFAULT_API_BASE,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        rate_per_second: float | None = 40.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = api_base.rstrip("/")
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng or random.Random()
        self._limiter = _RateLimiter(rate_per_second)
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                timeout=httpx.Timeout(30.0),
                transport=transport,
                headers=self._auth_headers(),
            )
            self._owns_client = True

    # -- public API ---------------------------------------------------------------

    def fetch_movie(self, tmdb_id: int) -> dict | None:
        """Return the enriched record for ``tmdb_id``, or ``None`` if TMDB has no such film."""
        params = {"append_to_response": _APPEND}
        # v3 key goes in the query; a v4 token rides in the Authorization header instead.
        if self._settings.tmdb_api_key is not None:
            params["api_key"] = self._settings.tmdb_api_key.get_secret_value()
        return self._get_with_retries(f"/movie/{tmdb_id}", params)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TMDBClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self._settings.tmdb_access_token
        if token is not None and token.get_secret_value():
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        return headers

    def _redact(self, text: str) -> str:
        return self._settings.redact(text)

    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        ceiling = min(_MAX_BACKOFF_SECONDS, _BASE_BACKOFF_SECONDS * (2**attempt))
        return ceiling * (0.5 + 0.5 * self._rng.random())

    def _get_with_retries(self, path: str, params: dict[str, str]) -> dict | None:
        url = f"{self._base_url}{path}"
        last_error: TMDBError | None = None
        for attempt in range(self._settings.max_retries + 1):
            self._limiter.acquire(sleep=self._sleep, monotonic=self._monotonic)
            retry_after: float | None = None
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as error:
                last_error = TMDBError(self._redact(f"Request failed: {error}"))
            else:
                outcome = self._handle_status(response)
                if not isinstance(outcome, TMDBError):
                    return outcome  # a dict on success, or None for 404
                last_error = outcome
                retry_after = _parse_retry_after(response)
            if attempt < self._settings.max_retries:
                self._sleep(self._backoff_seconds(attempt, retry_after))
        assert last_error is not None
        raise last_error

    def _handle_status(self, response: httpx.Response) -> dict | None | TMDBError:
        """Success → dict; 404 → None; retryable → a TMDBError to loop on; fatal → raised."""
        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError as error:
                return TMDBError(self._redact(f"Malformed JSON from TMDB: {error}"))
        if status == 404:
            return None
        if status in (401, 403):
            raise TMDBAuthError(self._redact(f"TMDB authentication failed ({status})."))
        if status in _RETRYABLE_STATUS:
            return TMDBError(self._redact(f"TMDB transient error ({status})."))
        raise TMDBError(self._redact(f"Unexpected TMDB status {status}."))


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# -- turning a TMDB record into catalog rows --------------------------------------


def _clean(value: object) -> object | None:
    """Normalise empty strings to NULL; pass everything else through."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def write_movie_details(
    conn: sqlite3.Connection, movie_id: int, data: dict, *, region: str
) -> None:
    """Write one enriched TMDB record into every relevant table for ``movie_id``.

    Join tables are cleared and rewritten so re-enriching a film is idempotent; dictionary
    tables (genres, keywords, people, providers, collections) accumulate with INSERT OR IGNORE.
    The caller owns the transaction and decides when to commit.
    """
    collection = data.get("belongs_to_collection")
    collection_id = None
    if isinstance(collection, dict) and collection.get("id") is not None:
        collection_id = int(collection["id"])
        conn.execute(
            "INSERT OR IGNORE INTO collections (collection_id, name) VALUES (?, ?)",
            (collection_id, collection.get("name") or ""),
        )

    conn.execute(
        """
        UPDATE movies SET
            original_title = ?, overview = ?, runtime = ?, original_language = ?,
            release_date = ?, poster_path = ?, popularity = ?, vote_average = ?,
            vote_count = ?, collection_id = ?, tmdb_fetched_at = ?
        WHERE movie_id = ?
        """,
        (
            _clean(data.get("original_title")),
            _clean(data.get("overview")),
            data.get("runtime"),
            _clean(data.get("original_language")),
            _clean(data.get("release_date")),
            _clean(data.get("poster_path")),
            data.get("popularity"),
            data.get("vote_average"),
            data.get("vote_count"),
            collection_id,
            _utcnow(),
            movie_id,
        ),
    )

    _write_genres(conn, movie_id, data.get("genres") or [])
    _write_keywords(conn, movie_id, (data.get("keywords") or {}).get("keywords") or [])
    _write_credits(conn, movie_id, data.get("credits") or {})
    _write_countries(conn, movie_id, data.get("production_countries") or [])
    _write_providers(conn, movie_id, data.get("watch/providers") or {}, region=region)


def mark_missing(conn: sqlite3.Connection, movie_id: int) -> None:
    """Stamp a film that TMDB returned 404 for, so it is not retried every run."""
    conn.execute("UPDATE movies SET tmdb_fetched_at = ? WHERE movie_id = ?", (_utcnow(), movie_id))


def _write_genres(conn: sqlite3.Connection, movie_id: int, genres: list) -> None:
    conn.execute("DELETE FROM movie_genres WHERE movie_id = ?", (movie_id,))
    for genre in genres:
        gid = genre.get("id")
        if gid is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO genres (genre_id, name) VALUES (?, ?)",
            (gid, genre.get("name") or ""),
        )
        conn.execute(
            "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
            (movie_id, gid),
        )


def _write_keywords(conn: sqlite3.Connection, movie_id: int, keywords: list) -> None:
    conn.execute("DELETE FROM movie_keywords WHERE movie_id = ?", (movie_id,))
    for keyword in keywords:
        kid = keyword.get("id")
        if kid is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO keywords (keyword_id, name) VALUES (?, ?)",
            (kid, keyword.get("name") or ""),
        )
        conn.execute(
            "INSERT OR IGNORE INTO movie_keywords (movie_id, keyword_id) VALUES (?, ?)",
            (movie_id, kid),
        )


def _write_credits(conn: sqlite3.Connection, movie_id: int, credits: dict) -> None:
    conn.execute("DELETE FROM credits WHERE movie_id = ?", (movie_id,))

    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 1_000_000))[:_MAX_CAST]
    for member in cast:
        pid = member.get("id")
        if pid is None:
            continue
        _ensure_person(conn, pid, member.get("name"))
        conn.execute(
            """
            INSERT OR IGNORE INTO credits
                (movie_id, person_id, credit_kind, job, department, character_name, billing)
            VALUES (?, ?, 'cast', '', NULL, ?, ?)
            """,
            (movie_id, pid, _clean(member.get("character")), member.get("order")),
        )

    for member in credits.get("crew") or []:
        job = member.get("job") or ""
        pid = member.get("id")
        if pid is None or job not in _KEEP_CREW_JOBS:
            continue
        _ensure_person(conn, pid, member.get("name"))
        conn.execute(
            """
            INSERT OR IGNORE INTO credits
                (movie_id, person_id, credit_kind, job, department, character_name, billing)
            VALUES (?, ?, 'crew', ?, ?, NULL, NULL)
            """,
            (movie_id, pid, job, _clean(member.get("department"))),
        )


def _ensure_person(conn: sqlite3.Connection, person_id: int, name: str | None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO people (person_id, name) VALUES (?, ?)",
        (person_id, name or ""),
    )


def _write_countries(conn: sqlite3.Connection, movie_id: int, countries: list) -> None:
    conn.execute("DELETE FROM movie_countries WHERE movie_id = ?", (movie_id,))
    for country in countries:
        code = country.get("iso_3166_1")
        if not code:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO movie_countries (movie_id, country_code) VALUES (?, ?)",
            (movie_id, code),
        )


def _write_providers(
    conn: sqlite3.Connection, movie_id: int, providers: dict, *, region: str
) -> None:
    conn.execute("DELETE FROM movie_watch_providers WHERE movie_id = ?", (movie_id,))
    region_data = (providers.get("results") or {}).get(region)
    if not region_data:
        return
    for monetization in ("flatrate", "rent", "buy", "ads", "free"):
        for provider in region_data.get(monetization) or []:
            pid = provider.get("provider_id")
            if pid is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO watch_providers (provider_id, name) VALUES (?, ?)",
                (pid, provider.get("provider_name") or ""),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO movie_watch_providers
                    (movie_id, provider_id, region, monetization)
                VALUES (?, ?, ?, ?)
                """,
                (movie_id, pid, region, monetization),
            )


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- orchestration ----------------------------------------------------------------


def select_targets(
    conn: sqlite3.Connection,
    *,
    scope: str = "measured",
    limit: int | None = None,
    force: bool = False,
) -> list[tuple[int, int]]:
    """The ``(movie_id, tmdb_id)`` pairs still needing enrichment, best-first.

    ``scope='measured'`` restricts to films with a genome vector (the searchable catalog);
    ``scope='all'`` takes every film with a TMDB id. Films already enriched are skipped unless
    ``force`` is set. Genome-covered films come first so a partial run enriches what matters.
    """
    # The clauses are fixed literals chosen here, never user input, so composing the WHERE by
    # join is safe. The only interpolated value, the limit, is coerced to int.
    clauses = ["tmdb_id IS NOT NULL"]
    if scope == "measured":
        clauses.append("genome_source = 'measured'")
    if not force:
        clauses.append("tmdb_fetched_at IS NULL")
    sql = (
        f"SELECT movie_id, tmdb_id FROM movies WHERE {' AND '.join(clauses)} "
        "ORDER BY (genome_source = 'measured') DESC, movie_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [(row["movie_id"], row["tmdb_id"]) for row in conn.execute(sql)]


def enrich_catalog(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    scope: str = "measured",
    region: str = "US",
    limit: int | None = None,
    force: bool = False,
    concurrency: int = 16,
    client: TMDBClient | None = None,
    console: Console | None = None,
    commit_every: int = 50,
) -> int:
    """Fetch and store TMDB details for the target films. Returns the number enriched.

    Fetches run concurrently; writes and commits happen here on the calling thread. A Ctrl-C
    commits what is done and returns, leaving the rest for the next (resumable) run.
    """
    console = console or Console()
    if not settings.has_tmdb_auth:
        raise TMDBAuthError(
            f"No TMDB credential set ({', '.join(('TMDB_API_KEY', 'TMDB_ACCESS_TOKEN'))})."
        )

    targets = select_targets(conn, scope=scope, limit=limit, force=force)
    if not targets:
        console.print("[dim]Nothing to enrich; every target film already has TMDB data.[/dim]")
        return 0

    owns_client = client is None
    client = client or TMDBClient(settings)
    enriched = 0
    processed = 0
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Enriching from TMDB", total=len(targets))
            executor = ThreadPoolExecutor(max_workers=concurrency)
            futures = {
                executor.submit(client.fetch_movie, tmdb_id): movie_id
                for movie_id, tmdb_id in targets
            }
            try:
                for future in as_completed(futures):
                    movie_id = futures[future]
                    data = future.result()  # TMDBAuthError here aborts the whole run
                    if data is None:
                        mark_missing(conn, movie_id)
                    else:
                        write_movie_details(conn, movie_id, data, region=region)
                        enriched += 1
                    processed += 1
                    progress.advance(task)
                    if processed % commit_every == 0:
                        conn.commit()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — committing progress (resumable).[/yellow]")
            finally:
                conn.commit()
                executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if owns_client:
            client.close()

    console.print(f"[green]Enriched {enriched:,} films from TMDB.[/green]")
    return enriched
