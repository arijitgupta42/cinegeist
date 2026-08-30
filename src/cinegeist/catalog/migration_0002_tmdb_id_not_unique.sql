-- Migration to user_version 2: tmdb_id is not unique.
--
-- The real MovieLens links.csv maps several different movieIds to the same tmdbId — alternate
-- or duplicate entries for what TMDB considers one film. The MovieLens movie_id is our real key
-- (the genome is keyed on it), and two such rows can carry different genome vectors, so we keep
-- every movie_id and simply allow the tmdb_id to repeat. Relax the unique index to a plain one;
-- movie_id stays the primary key.
DROP INDEX IF EXISTS idx_movies_tmdb;
CREATE INDEX IF NOT EXISTS idx_movies_tmdb ON movies (tmdb_id) WHERE tmdb_id IS NOT NULL;
