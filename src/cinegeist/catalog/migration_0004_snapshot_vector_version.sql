-- CineGeist — migration to user_version 4: record which masking scheme a snapshot's vector used.
--
-- The taste centroid now zeroes the "non-content" genome tags — reception, curation, awards, and
-- pure quality verdicts (see catalog/excluded_tags.py and plan.md session 8) — so that two films
-- sharing `imdb top 250` or `masterpiece` are no longer pulled together or cited as a match. That
-- masking happens in profile/update.py when the centroid is computed.
--
-- profile_snapshots caches the decayed centroid. A snapshot written before the masking existed
-- holds the *un*-masked vector, yet its event_count still matches the live log, so a reader would
-- happily reuse it and keep steering on the excluded tags. To stop that, the cache now records the
-- masking scheme id (profile/update.py MASK_VERSION) that produced its vector; a reader reuses a
-- snapshot only when that id matches the current one, and recomputes anything older.
--
-- profile_snapshots is a pure derived cache — safe to drop entirely and rebuild on next read — so
-- we recreate it with the new column. That both adds vector_version and discards every pre-masking
-- vector in one step. DROP + CREATE ... IF NOT EXISTS is idempotent, so an interrupted replay is
-- clean, matching the runner in db.py that only advances user_version after a whole script runs.
DROP TABLE IF EXISTS profile_snapshots;
CREATE TABLE IF NOT EXISTS profile_snapshots (
    user_id        TEXT PRIMARY KEY,
    computed_at    TEXT NOT NULL,              -- ISO 8601 UTC; the "now" the decay was anchored to
    event_count    INTEGER NOT NULL,           -- events behind this snapshot; a cheap staleness check
    total_weight   REAL NOT NULL,              -- Σ|w_i| at computed_at; the evidence mass / confidence
    genome_vector  BLOB NOT NULL,              -- float32[n_tags] little-endian; the decayed centroid
    vector_version INTEGER NOT NULL DEFAULT 0  -- masking scheme id; a reader rejects a stale version
);
