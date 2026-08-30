-- CineGeist catalog schema — migration to user_version 1.
--
-- This file is the single source of truth for the catalog's shape. Every statement is
-- idempotent (CREATE ... IF NOT EXISTS), so re-running a partially-applied migration after
-- a Ctrl-C is safe. The migration runner in db.py only bumps PRAGMA user_version once the
-- whole script has executed, so an interrupted run stays at the previous version and gets
-- replayed cleanly next time.
--
-- What lives where:
--   * the dense tag-genome MATRIX lives in data/genome.npy (a numpy memmap), NOT here — a
--     full-catalog cosine scan is one matmul, and 14M floats have no business in SQLite.
--     movies.genome_row is this table's link into that matrix; genome_tags.position is the
--     column order. See catalog/genome.py.
--   * everything else — metadata, facets, credits, the eventual user profile — is SQLite.
--
-- Columns filled by the MovieLens ingest (session 2, PR 2) vs. the TMDB enrichment (PR 3)
-- are marked below. TMDB columns are NULL until a film is enriched.

-- One row per MovieLens film we ingest. movie_id is the MovieLens movieId throughout the
-- app; it is our stable join key. tmdb_id / imdb_id come from MovieLens links.csv.
CREATE TABLE IF NOT EXISTS movies (
    movie_id            INTEGER PRIMARY KEY,       -- MovieLens movieId
    imdb_id             TEXT,                      -- 'tt0111161'; from links.csv, may be NULL
    tmdb_id             INTEGER,                   -- TMDB id; from links.csv, may be NULL
    title               TEXT NOT NULL,             -- MovieLens title, e.g. "Toy Story (1995)"
    clean_title         TEXT,                      -- title with the trailing "(year)" removed
    year                INTEGER,                   -- year parsed from the title, may be NULL

    -- TMDB enrichment (PR 3). NULL until the film has been fetched.
    original_title      TEXT,
    overview            TEXT,
    runtime             INTEGER,                   -- minutes
    original_language   TEXT,                      -- ISO 639-1, e.g. 'en'
    release_date        TEXT,                      -- ISO 'YYYY-MM-DD'
    poster_path         TEXT,                      -- TMDB image path, e.g. '/abc.jpg'
    popularity          REAL,
    vote_average        REAL,
    vote_count          INTEGER,
    collection_id       INTEGER,                   -- TMDB collection membership, NULL if none

    -- Genome linkage. genome_row is the row index into data/genome.npy; NULL means we have
    -- no vector for this film yet. genome_source records where that vector came from.
    genome_row          INTEGER,
    genome_source       TEXT NOT NULL DEFAULT 'none'
                          CHECK (genome_source IN ('none', 'measured', 'predicted')),

    -- Bookkeeping for the resumable TMDB crawl: ISO timestamp of the last successful fetch.
    tmdb_fetched_at     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_tmdb
    ON movies (tmdb_id) WHERE tmdb_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_movies_imdb ON movies (imdb_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_genome_row
    ON movies (genome_row) WHERE genome_row IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_movies_year ON movies (year);
-- Speeds up the resumable crawl's "which films still need TMDB?" scan.
CREATE INDEX IF NOT EXISTS idx_movies_unfetched
    ON movies (tmdb_fetched_at) WHERE tmdb_fetched_at IS NULL AND tmdb_id IS NOT NULL;

-- The tag genome's tag dictionary (MovieLens genome-tags.csv). `position` is the 0-based
-- column index of this tag in data/genome.npy — the mapping that lets genome.py turn a tag
-- name into a matrix column. Assigned once, in tag_id order, when the genome is built.
CREATE TABLE IF NOT EXISTS genome_tags (
    tag_id      INTEGER PRIMARY KEY,               -- MovieLens tagId
    position    INTEGER NOT NULL UNIQUE,           -- 0-based column index into genome.npy
    name        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genome_tags_name ON genome_tags (name);

-- TMDB genres (a small controlled vocabulary) and the film↔genre join.
CREATE TABLE IF NOT EXISTS genres (
    genre_id    INTEGER PRIMARY KEY,               -- TMDB genre id
    name        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id    INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    genre_id    INTEGER NOT NULL REFERENCES genres (genre_id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);
CREATE INDEX IF NOT EXISTS idx_movie_genres_genre ON movie_genres (genre_id);

-- TMDB keywords (free-text tags, ~10–40 per film) and the join. These are the coarse
-- fallback signal for films the tag genome doesn't cover.
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id  INTEGER PRIMARY KEY,               -- TMDB keyword id
    name        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS movie_keywords (
    movie_id    INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    keyword_id  INTEGER NOT NULL REFERENCES keywords (keyword_id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_movie_keywords_keyword ON movie_keywords (keyword_id);

-- Cast and crew, deduplicated into a people dictionary plus a credits join. One person can
-- hold several crew jobs on one film (writer and director), so `job` is part of the key; it
-- defaults to '' rather than NULL so that the primary key actually dedups cast rows (SQLite
-- treats every NULL as distinct, which would let duplicate cast credits slip in).
CREATE TABLE IF NOT EXISTS people (
    person_id   INTEGER PRIMARY KEY,               -- TMDB person id
    name        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS credits (
    movie_id        INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    person_id       INTEGER NOT NULL REFERENCES people (person_id) ON DELETE CASCADE,
    credit_kind     TEXT NOT NULL CHECK (credit_kind IN ('cast', 'crew')),
    job             TEXT NOT NULL DEFAULT '',       -- crew job, e.g. 'Director'; '' for cast
    department      TEXT,                            -- crew department; NULL for cast
    character_name  TEXT,                            -- cast character; NULL for crew
    billing         INTEGER,                         -- cast order (0 = top-billed); NULL for crew
    PRIMARY KEY (movie_id, person_id, credit_kind, job)
);
CREATE INDEX IF NOT EXISTS idx_credits_person ON credits (person_id);
-- The common lookup: "who directed this?" and other job-scoped queries.
CREATE INDEX IF NOT EXISTS idx_credits_movie_job ON credits (movie_id, job);

-- Production countries (ISO 3166-1 alpha-2), one row per film per country.
CREATE TABLE IF NOT EXISTS movie_countries (
    movie_id        INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    country_code    TEXT NOT NULL,                   -- ISO 3166-1 alpha-2, e.g. 'US'
    PRIMARY KEY (movie_id, country_code)
);

-- Regional streaming availability (TMDB /watch/providers). monetization distinguishes
-- subscription ('flatrate') from 'rent' / 'buy' / 'ads' / 'free'.
CREATE TABLE IF NOT EXISTS watch_providers (
    provider_id INTEGER PRIMARY KEY,               -- TMDB provider id
    name        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS movie_watch_providers (
    movie_id        INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    provider_id     INTEGER NOT NULL REFERENCES watch_providers (provider_id) ON DELETE CASCADE,
    region          TEXT NOT NULL,                   -- ISO 3166-1 alpha-2, e.g. 'US'
    monetization    TEXT NOT NULL,                   -- flatrate | rent | buy | ads | free
    PRIMARY KEY (movie_id, provider_id, region, monetization)
);
CREATE INDEX IF NOT EXISTS idx_movie_providers_region ON movie_watch_providers (region);

-- TMDB collections (franchises). movies.collection_id points here; no FK, so a film can be
-- ingested before its collection row exists without tripping foreign_keys.
CREATE TABLE IF NOT EXISTS collections (
    collection_id   INTEGER PRIMARY KEY,           -- TMDB collection id
    name            TEXT NOT NULL
);

-- A tiny key/value scratchpad for the build pipeline: stage timestamps, the genome memmap's
-- shape and dtype, the TMDB /changes cursor for incremental refresh, and so on. Reading and
-- writing progress here is what makes `make catalog` resumable across stages.
CREATE TABLE IF NOT EXISTS build_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
