"""Unit tests for TMDB enrichment: the client, the response writers, and the orchestrator.

All HTTP is mocked. The client's retry, rate-limit, and auth branches are driven with injected
clocks so nothing sleeps for real and nothing touches the network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from cinegeist.catalog import db
from cinegeist.catalog.sources import tmdb
from cinegeist.config import Settings


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"tmdb_api_key": SecretStr("test-tmdb-key"), "max_retries": 2}
    base.update(overrides)
    return Settings(**base)


def sample_movie(tmdb_id: int = 862, *, cast_count: int = 20) -> dict:
    cast = [
        {"id": 1000 + i, "name": f"Actor {i}", "character": f"Role {i}", "order": i}
        for i in range(cast_count)
    ]
    return {
        "id": tmdb_id,
        "original_title": "Toy Story",
        "overview": "A cowboy doll is threatened by a spaceman.",
        "runtime": 81,
        "original_language": "en",
        "release_date": "1995-11-22",
        "poster_path": "/abc.jpg",
        "popularity": 21.9,
        "vote_average": 7.9,
        "vote_count": 12000,
        "belongs_to_collection": {"id": 10, "name": "Toy Story Collection"},
        "genres": [{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}],
        "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
        "keywords": {"keywords": [{"id": 931, "name": "jealousy"}, {"id": 4290, "name": "toy"}]},
        "credits": {
            "cast": cast,
            "crew": [
                {"id": 7879, "name": "John Lasseter", "job": "Director", "department": "Directing"},
                {"id": 12891, "name": "Joss Whedon", "job": "Screenplay", "department": "Writing"},
                {"id": 9999, "name": "A Grip", "job": "Grip", "department": "Crew"},  # dropped
            ],
        },
        "watch/providers": {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 8, "provider_name": "Netflix"}],
                    "rent": [{"provider_id": 2, "provider_name": "Apple TV"}],
                },
                "GB": {"flatrate": [{"provider_id": 9, "provider_name": "Prime"}]},
            }
        },
    }


def _client(handler, **kwargs: object) -> tmdb.TMDBClient:
    return tmdb.TMDBClient(
        make_settings(),
        transport=httpx.MockTransport(handler),
        rate_per_second=None,
        sleep=lambda _s: None,
        **kwargs,
    )


# -- client -----------------------------------------------------------------------


def test_fetch_movie_parses_and_requests_append() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/movie/862")
        assert request.url.params["append_to_response"] == "keywords,credits,watch/providers"
        assert request.url.params["api_key"] == "test-tmdb-key"
        return httpx.Response(200, json=sample_movie())

    data = _client(handler).fetch_movie(862)
    assert data["runtime"] == 81


def test_fetch_movie_returns_none_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status_code": 34})

    assert _client(handler).fetch_movie(1) is None


def test_fetch_movie_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=sample_movie())

    data = _client(handler).fetch_movie(862)
    assert calls["n"] == 2
    assert data["id"] == 862


def test_fetch_movie_raises_on_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"status_code": 7})

    with pytest.raises(tmdb.TMDBAuthError):
        _client(handler).fetch_movie(862)


def test_error_message_redacts_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    try:
        _client(handler).fetch_movie(862)
    except tmdb.TMDBError as error:
        assert "test-tmdb-key" not in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a TMDBError")


def test_uses_bearer_token_when_no_api_key() -> None:
    settings = Settings(tmdb_access_token=SecretStr("v4-token"))
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["api_key"] = request.url.params.get("api_key")
        return httpx.Response(200, json=sample_movie())

    client = tmdb.TMDBClient(settings, transport=httpx.MockTransport(handler), rate_per_second=None)
    client.fetch_movie(862)
    assert seen["auth"] == "Bearer v4-token"
    assert seen["api_key"] is None  # token path never puts a secret in the URL


# -- rate limiter -----------------------------------------------------------------


def test_rate_limiter_spaces_requests() -> None:
    slept: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = tmdb._RateLimiter(per_second=5.0)  # 0.2s apart
    limiter.acquire(sleep=sleep, monotonic=lambda: now[0])  # first is free
    limiter.acquire(sleep=sleep, monotonic=lambda: now[0])  # second waits a full interval
    assert slept == [pytest.approx(0.2)]


# -- writers ----------------------------------------------------------------------


def _catalog_with_movie(
    tmp_path: Path, movie_id: int = 1, tmdb_id: int = 862
) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "c.db")
    db.migrate(conn)
    conn.execute(
        "INSERT INTO movies (movie_id, tmdb_id, title, genome_source) "
        "VALUES (?, ?, 'X (1)', 'measured')",
        (movie_id, tmdb_id),
    )
    return conn


def test_write_movie_details_fills_every_table(tmp_path: Path) -> None:
    conn = _catalog_with_movie(tmp_path)
    tmdb.write_movie_details(conn, 1, sample_movie(), region="US")

    movie = conn.execute("SELECT * FROM movies WHERE movie_id = 1").fetchone()
    assert movie["runtime"] == 81
    assert movie["original_language"] == "en"
    assert movie["collection_id"] == 10
    assert movie["vote_average"] == 7.9
    assert movie["tmdb_fetched_at"] is not None

    assert conn.execute("SELECT name FROM collections WHERE collection_id = 10").fetchone()[0]
    assert conn.execute("SELECT COUNT(*) FROM movie_genres WHERE movie_id = 1").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM movie_keywords WHERE movie_id = 1").fetchone()[0] == 2
    assert (
        conn.execute("SELECT COUNT(*) FROM movie_countries WHERE movie_id = 1").fetchone()[0] == 1
    )


def test_write_movie_details_caps_cast_and_filters_crew(tmp_path: Path) -> None:
    conn = _catalog_with_movie(tmp_path)
    tmdb.write_movie_details(conn, 1, sample_movie(cast_count=20), region="US")

    cast = conn.execute(
        "SELECT COUNT(*) FROM credits WHERE movie_id = 1 AND credit_kind = 'cast'"
    ).fetchone()[0]
    assert cast == 15  # capped from 20
    crew_jobs = {
        row[0]
        for row in conn.execute(
            "SELECT job FROM credits WHERE movie_id = 1 AND credit_kind = 'crew'"
        )
    }
    assert crew_jobs == {"Director", "Screenplay"}  # the Grip was dropped


def test_write_movie_details_only_stores_the_requested_region(tmp_path: Path) -> None:
    conn = _catalog_with_movie(tmp_path)
    tmdb.write_movie_details(conn, 1, sample_movie(), region="US")
    providers = conn.execute(
        "SELECT region, monetization FROM movie_watch_providers WHERE movie_id = 1"
    ).fetchall()
    assert {(p["region"], p["monetization"]) for p in providers} == {
        ("US", "flatrate"),
        ("US", "rent"),
    }


def test_write_movie_details_is_idempotent(tmp_path: Path) -> None:
    conn = _catalog_with_movie(tmp_path)
    tmdb.write_movie_details(conn, 1, sample_movie(), region="US")
    tmdb.write_movie_details(conn, 1, sample_movie(), region="US")  # re-enrich
    assert conn.execute("SELECT COUNT(*) FROM movie_genres WHERE movie_id = 1").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM movie_keywords WHERE movie_id = 1").fetchone()[0] == 2


# -- target selection -------------------------------------------------------------


def _seed(conn: sqlite3.Connection, movie_id: int, **cols: object) -> None:
    columns = ["movie_id", "title", *cols.keys()]
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO movies ({', '.join(columns)}) VALUES ({placeholders})",
        (movie_id, f"M{movie_id} (1)", *cols.values()),
    )


def test_select_targets_scope_and_order(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "c.db")
    db.migrate(conn)
    _seed(conn, 1, tmdb_id=100, genome_source="measured")
    _seed(conn, 2, tmdb_id=200, genome_source="none")  # unmeasured
    _seed(conn, 3, tmdb_id=None, genome_source="measured")  # no tmdb id
    _seed(conn, 4, tmdb_id=400, genome_source="measured", tmdb_fetched_at="2020-01-01T00:00:00Z")

    measured = tmdb.select_targets(conn, scope="measured")
    assert measured == [(1, 100)]  # 2 unmeasured, 3 no id, 4 already fetched

    every = tmdb.select_targets(conn, scope="all")
    assert every == [(1, 100), (2, 200)]  # measured first, then by movie_id

    forced = tmdb.select_targets(conn, scope="all", force=True)
    assert (4, 400) in forced  # force ignores tmdb_fetched_at

    assert tmdb.select_targets(conn, scope="all", limit=1) == [(1, 100)]


# -- orchestration ----------------------------------------------------------------


def _enrich_handler(missing: set[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        tmdb_id = int(request.url.path.rsplit("/", 1)[-1])
        if tmdb_id in missing:
            return httpx.Response(404, json={"status_code": 34})
        return httpx.Response(200, json=sample_movie(tmdb_id))

    return handler


def test_enrich_catalog_end_to_end_and_resumes(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "c.db")
    db.migrate(conn)
    _seed(conn, 1, tmdb_id=862, genome_source="measured")
    _seed(conn, 2, tmdb_id=593, genome_source="measured")
    _seed(conn, 3, tmdb_id=777, genome_source="measured")  # TMDB will 404 this one
    conn.commit()

    client = _client(_enrich_handler(missing={777}))
    enriched = tmdb.enrich_catalog(conn, make_settings(), client=client, concurrency=4)
    assert enriched == 2  # films 1 and 2; film 3 was missing

    # Every target is stamped, including the 404 (so it is not retried forever).
    unfetched = conn.execute(
        "SELECT COUNT(*) FROM movies WHERE tmdb_fetched_at IS NULL"
    ).fetchone()[0]
    assert unfetched == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM movie_genres").fetchone()[0] == 4
    )  # 2 films × 2 genres

    # A second run has nothing left to do.
    again = tmdb.enrich_catalog(conn, make_settings(), client=_client(_enrich_handler(set())))
    assert again == 0


def test_enrich_catalog_requires_a_credential(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "c.db")
    db.migrate(conn)
    with pytest.raises(tmdb.TMDBAuthError):
        tmdb.enrich_catalog(conn, Settings())  # no TMDB auth
