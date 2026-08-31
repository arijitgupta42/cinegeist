-- CineGeist profile store — migration to user_version 3.
--
-- The taste profile is an append-only event log, never mutated in place (plan.md §4.2,
-- CLAUDE.md hard rule 7). Every reaction the user gives — a film they loved, one they
-- bounced off, a this-or-that choice, an answer to an axis question, a session constraint —
-- lands here as one immutable row carrying the user's own words. The profile the recommender
-- actually uses is a *derived* view: a decayed, weighted centroid recomputed from these rows.
--
-- Three properties fall out of this for free:
--   * explainability — the evidence string is right there next to the number it produced;
--   * correctability — `cinegeist profile forget <id>` deletes one wrong row and nothing else;
--   * auditability — the user can read exactly what the system thinks it knows, in their words.
--
-- Every statement is idempotent (CREATE ... IF NOT EXISTS), so an interrupted migration is
-- replayed cleanly — the runner in db.py only advances user_version after the whole script runs.

-- One row per reaction. Immutable once written; we only ever INSERT and (on an explicit
-- forget/reset) DELETE. `subject` is polymorphic — a movie id, a genome tag id, or a facet
-- key — disambiguated by `subject_kind`, so it is stored as TEXT and parsed back by the
-- reader. There is deliberately no foreign key to movies: an event is evidence about the
-- user and must outlive a catalog rebuild that happens to drop or renumber a film.
CREATE TABLE IF NOT EXISTS preference_events (
    id            INTEGER PRIMARY KEY,
    user_id       TEXT NOT NULL DEFAULT 'default',
    ts            TEXT NOT NULL,             -- ISO 8601 UTC; when the reaction happened
    session_id    TEXT,                      -- groups a conversation's events; NULL if unknown

    -- What kind of reaction this is. Drives how the event feeds the profile:
    --   liked_movie / disliked_movie / pair_choice / post_watch_feedback  → a movie vector
    --   axis_answer                                                       → a single tag axis
    --   constraint                                                        → a facet/filter, not taste
    kind          TEXT NOT NULL CHECK (kind IN (
                      'liked_movie', 'disliked_movie', 'pair_choice',
                      'axis_answer', 'constraint', 'post_watch_feedback')),

    -- What the reaction is *about*. subject_kind says how to read subject:
    --   'movie' → subject is a movies.movie_id      (contributes that film's genome row)
    --   'tag'   → subject is a genome_tags.tag_id   (contributes a one-hot on that axis)
    --   'facet' → subject is a facet key, e.g. 'max_runtime' (a constraint, no genome vector)
    subject_kind  TEXT NOT NULL CHECK (subject_kind IN ('movie', 'tag', 'facet')),
    subject       TEXT NOT NULL,

    value         REAL NOT NULL,             -- -1.0 .. +1.0: direction and strength of the signal
    weight        REAL NOT NULL DEFAULT 1.0, -- initial confidence, before time decay is applied
    evidence      TEXT                       -- the user's verbatim words; NULL for a bare click
);
CREATE INDEX IF NOT EXISTS idx_pref_events_user ON preference_events (user_id, ts);
CREATE INDEX IF NOT EXISTS idx_pref_events_session ON preference_events (user_id, session_id);

-- The derived taste vector, cached so a turn doesn't replay the whole log. The centroid is a
-- ratio of sums, and uniform time decay multiplies every term by the same factor, so it cancels
-- top and bottom: the cached genome_vector stays exact until the *event set* changes, however
-- much time passes. Only total_weight (the evidence mass, our confidence signal) decays with
-- time, and it does so by one scalar factor we can apply on read. So invalidation is simply
-- "the events changed": store.py deletes this row on every append, forget, and reset, and a
-- reader trusts a snapshot only when its event_count still matches the live count.
CREATE TABLE IF NOT EXISTS profile_snapshots (
    user_id       TEXT PRIMARY KEY,
    computed_at   TEXT NOT NULL,             -- ISO 8601 UTC; the "now" the decay was anchored to
    event_count   INTEGER NOT NULL,          -- events behind this snapshot; a cheap staleness check
    total_weight  REAL NOT NULL,             -- Σ|w_i| at computed_at; the evidence mass / confidence
    genome_vector BLOB NOT NULL              -- float32[n_tags] little-endian; the decayed centroid
);
