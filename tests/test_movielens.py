"""Unit tests for the MovieLens source: title parsing, verification, readers, resumable download.

The download tests drive a mocked transport — no bytes leave the machine. The lone real-network
test is marked and excluded from CI.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest

from cinegeist.catalog.sources import movielens


def test_parse_title_with_year() -> None:
    assert movielens.parse_title("Toy Story (1995)") == ("Toy Story", 1995)
    assert movielens.parse_title("Amélie (2001)") == ("Amélie", 2001)


def test_parse_title_without_year() -> None:
    assert movielens.parse_title("Untitled Short") == ("Untitled Short", None)


def test_parse_title_keeps_interior_parentheses() -> None:
    # Only a trailing (YYYY) is the year; a parenthetical in the body is left alone.
    assert movielens.parse_title("Léon: The Professional (a.k.a. Leon) (1994)") == (
        "Léon: The Professional (a.k.a. Leon)",
        1994,
    )


def test_verify_archive_accepts_the_real_shape(movielens_archive: Path) -> None:
    assert movielens.verify_archive(movielens_archive) is True


def test_verify_archive_rejects_missing_file(tmp_path: Path) -> None:
    assert movielens.verify_archive(tmp_path / "nope.zip") is False


def test_verify_archive_rejects_truncation(movielens_archive: Path, tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(movielens_archive.read_bytes()[:50])  # lop off the central directory
    assert movielens.verify_archive(truncated) is False


def test_verify_archive_rejects_a_missing_member(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete, "w") as archive:
        archive.writestr("ml-latest/movies.csv", "movieId,title,genres\n")
        # links, genome-tags, genome-scores all absent
    assert movielens.verify_archive(incomplete) is False


def test_iter_movies(movielens_archive: Path) -> None:
    rows = list(movielens.iter_movies(movielens_archive))
    assert rows[0] == (1, "Toy Story (1995)", "Toy Story", 1995, "Adventure|Animation|Children")
    assert rows[2] == (3, "Untitled Short", "Untitled Short", None, "Documentary")


def test_iter_links_restores_imdb_prefix_and_handles_blanks(movielens_archive: Path) -> None:
    rows = {
        movie_id: (imdb, tmdb) for movie_id, imdb, tmdb in movielens.iter_links(movielens_archive)
    }
    assert rows[1] == ("tt0114709", 862)
    assert rows[3] == (None, None)


def test_iter_genome_tags(movielens_archive: Path) -> None:
    assert list(movielens.iter_genome_tags(movielens_archive)) == [
        (1, "animation"),
        (2, "cerebral"),
        (3, "space"),
    ]


def test_iter_genome_scores(movielens_archive: Path) -> None:
    scores = list(movielens.iter_genome_scores(movielens_archive))
    assert (1, 1, 0.9) in scores
    assert (
        999,
        3,
        0.5,
    ) in scores  # the reader passes everything through; filtering is the build's job
    assert len(scores) == 9


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_fresh(tmp_path: Path, movielens_archive: Path) -> None:
    content = movielens_archive.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("range") is None  # nothing partial yet
        return httpx.Response(200, content=content, headers={"Content-Length": str(len(content))})

    dest = movielens.download_archive(
        tmp_path / "dl", url="http://example/ml-latest.zip", client=_mock_client(handler)
    )
    assert dest.read_bytes() == content
    assert movielens.verify_archive(dest)
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumes_from_a_partial_file(tmp_path: Path, movielens_archive: Path) -> None:
    content = movielens_archive.read_bytes()
    split = len(content) // 2
    dest_dir = tmp_path / "dl"
    dest_dir.mkdir()
    (dest_dir / (movielens.ARCHIVE_NAME + ".part")).write_bytes(content[:split])

    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers.get("range")
        return httpx.Response(
            206,
            content=content[split:],
            headers={
                "Content-Length": str(len(content) - split),
                "Content-Range": f"bytes {split}-{len(content) - 1}/{len(content)}",
            },
        )

    dest = movielens.download_archive(
        dest_dir, url="http://example/ml-latest.zip", client=_mock_client(handler)
    )
    assert seen["range"] == f"bytes={split}-"  # it asked to resume
    assert dest.read_bytes() == content
    assert movielens.verify_archive(dest)


def test_download_restarts_when_server_ignores_range(
    tmp_path: Path, movielens_archive: Path
) -> None:
    content = movielens_archive.read_bytes()
    dest_dir = tmp_path / "dl"
    dest_dir.mkdir()
    (dest_dir / (movielens.ARCHIVE_NAME + ".part")).write_bytes(b"stale partial junk")

    def handler(request: httpx.Request) -> httpx.Response:
        # Server ignores the Range header and replies with the whole file.
        return httpx.Response(200, content=content, headers={"Content-Length": str(len(content))})

    dest = movielens.download_archive(
        dest_dir, url="http://example/ml-latest.zip", client=_mock_client(handler)
    )
    assert dest.read_bytes() == content  # the stale prefix was discarded, not appended to


def test_download_reuses_a_valid_archive_without_the_network(
    tmp_path: Path, movielens_archive: Path
) -> None:
    dest_dir = tmp_path / "dl"
    dest_dir.mkdir()
    (dest_dir / movielens.ARCHIVE_NAME).write_bytes(movielens_archive.read_bytes())

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("download should not touch the network when the archive is valid")

    dest = movielens.download_archive(
        dest_dir, url="http://example/ml-latest.zip", client=_mock_client(handler)
    )
    assert dest == dest_dir / movielens.ARCHIVE_NAME


@pytest.mark.network
def test_grouplens_url_is_reachable() -> None:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(movielens.DEFAULT_URL, headers={"Range": "bytes=0-1023"})
    assert response.status_code in (200, 206)
