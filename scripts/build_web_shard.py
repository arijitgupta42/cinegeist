"""Build the browser-demo shard from the local catalog and write it into ``web/public/shard/``.

This is the thin orchestrator behind ``make web-shard`` (plan.md §8.3): it opens ``data/`` and the
genome memmap, hands them to :mod:`cinegeist.webshard.build`, writes ``shard.json`` and
``shard.bin``, and reports the gzipped size against the 400 KB budget. It needs the full catalog,
so it runs locally and by hand, never in CI — the shard it produces is committed, and a CI test
checks that committed artifact instead of rebuilding it.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cinegeist.catalog import db, genome  # noqa: E402
from cinegeist.config import data_dir  # noqa: E402
from cinegeist.webshard.build import build_probes, build_shard  # noqa: E402

OUT_DIR = ROOT / "web" / "public" / "shard"
SIZE_BUDGET_KB = 400


def _write_lf(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    db_path = db.default_db_path()
    genome_path = genome.default_genome_path(data_dir())
    if not db_path.exists() or not genome_path.exists():
        raise SystemExit(
            f"catalog not found ({db_path} / {genome_path}); run `make catalog` first."
        )

    conn = db.open_catalog(db_path)
    matrix = genome.load_genome(genome_path)
    print(f"catalog: {matrix.shape[0]} genome films x {matrix.shape[1]} tags")

    build = build_shard(conn, matrix)
    probes = build_probes(conn, matrix)
    conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Compact on purpose: these are generated artifacts the demo parses, not files anyone hand-edits
    # or reads a git diff of line by line. Compact keeps them small and signals "regenerate, don't
    # edit". ensure_ascii=False stores UTF-8 directly (smaller than \uXXXX for accented titles).
    manifest_json = json.dumps(build.manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    probes_json = json.dumps(probes, ensure_ascii=False, separators=(",", ":")) + "\n"
    _write_lf(OUT_DIR / "shard.json", manifest_json)
    (OUT_DIR / "shard.bin").write_bytes(build.binary)
    _write_lf(OUT_DIR / "probes.json", probes_json)

    json_bytes = manifest_json.encode("utf-8")
    probes_bytes = probes_json.encode("utf-8")
    shard_gz = len(gzip.compress(build.binary, 9)) + len(gzip.compress(json_bytes, 9))
    probes_gz = len(gzip.compress(probes_bytes, 9))
    print(f"films: {build.manifest['n_films']}  components: {build.manifest['n_components']}")
    print(f"shard.bin   {len(build.binary) / 1024:7.1f} KB")
    print(f"shard.json  {len(json_bytes) / 1024:7.1f} KB")
    print(f"probes.json {len(probes_bytes) / 1024:7.1f} KB  ({len(probes['probes'])} probes)")
    print(f"gzipped: shard {shard_gz / 1024:.1f} KB (budget {SIZE_BUDGET_KB}) + probes "
          f"{probes_gz / 1024:.1f} KB")
    if shard_gz / 1024 > SIZE_BUDGET_KB:
        raise SystemExit(f"shard is over the {SIZE_BUDGET_KB} KB gzipped budget")
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/shard.json + shard.bin + probes.json")


if __name__ == "__main__":
    main()
