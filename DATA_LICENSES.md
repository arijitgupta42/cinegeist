# Data licences and attribution

cinegeist does not redistribute any dataset. The catalog is downloaded and built on
your machine at build time (`make catalog`). This file records where the data comes
from and how it is licensed, and it must stay accurate.

## MovieLens (GroupLens Research)

The tag genome and movie metadata come from the MovieLens `ml-latest` dataset.
It is provided for research and personal use; see the dataset's own `README` and
usage licence at <https://grouplens.org/datasets/movielens/>. When you build the
catalog you accept those terms.

Please cite:

- F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
  and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
  Article 19. <https://doi.org/10.1145/2827872>
- Jesse Vig, Shilad Sen, and John Riedl. 2012. *The Tag Genome: Encoding Community
  Knowledge to Support Novel Interaction.* ACM Transactions on Interactive
  Intelligent Systems (TiiS) 2, 3, Article 13. <https://doi.org/10.1145/2362394.2362395>

## TMDB

Coverage, freshness, and structured facets come from The Movie Database (TMDB) API,
used under its non-commercial terms with attribution. A key is required and is read
from the environment; it is never committed or logged.

> This product uses the TMDB API but is not endorsed or certified by TMDB.

See <https://www.themoviedb.org/documentation/api/terms-of-use>.

## What ships in this repository

Nothing from either dataset. `data/`, `*.db`, and `*.npy` are gitignored. The small
`tests/fixtures/` catalog (added in a later session) contains a few hundred films with
real genome vectors, kept for offline testing under the same MovieLens research terms.
