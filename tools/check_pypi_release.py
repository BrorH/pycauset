"""Guards for publishing to PyPI without burning the project's storage quota.

PyPI caps the *total* size of a project (default 10 GB, every version ever
uploaded counts, and yanking does not free space; only deleting a version does).
One full PyCauset release is ~270 MB (18 wheels + sdist), so the quota holds
dozens of real releases but is exhausted quickly if dev builds are uploaded.

Two subcommands, both used by `.github/workflows/publish.yml`:

    python tools/check_pypi_release.py preflight --tag v0.7.2
        Fails fast (before ~25 min of wheel builds) if the tag is not a plain
        release version (vX.Y.Z), or if PyPI does not have room for one more
        release. Also prints the current PyPI usage.

    python tools/check_pypi_release.py verify-dist dist --tag v0.7.2
        Run on the built artifacts right before upload. Fails if any file is a
        dev/local/other-version build, or if the upload is unexpectedly large.

Run `preflight` locally (without --tag) to just see the current usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = "pycauset"

# PyPI's default per-project total-size limit ("10 GB", decimal).
PROJECT_LIMIT_BYTES = 10 * 1000**3
# Refuse to start a release that would push usage above this fraction of the limit.
MAX_USAGE_FRACTION = 0.95
# Warn (but continue) above this fraction, so cleanup happens before it is urgent.
WARN_USAGE_FRACTION = 0.75
# Used when no previous release exists to estimate from.
DEFAULT_RELEASE_BYTES = 300 * 1000**2
# A single release larger than this is almost certainly a mistake (e.g. debug
# builds or duplicated wheels); the current size is ~270 MB.
MAX_RELEASE_BYTES = 450 * 1000**2

RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _gb(n: int) -> str:
    return f"{n / 1000**3:.2f} GB"


def _mb(n: int) -> str:
    return f"{n / 1000**2:.1f} MB"


def _summary(text: str) -> None:
    """Print, and also append to the GitHub Actions job summary when available."""
    print(text)
    import os

    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def version_from_tag(tag: str) -> str:
    m = RELEASE_TAG.match(tag.strip())
    if not m:
        raise SystemExit(
            f"ERROR: tag {tag!r} is not a release tag of the form vX.Y.Z "
            "(e.g. v0.7.2). Dev, rc and other tags are never published to PyPI."
        )
    return ".".join(m.groups())


def fetch_pypi_releases() -> dict[str, list[dict]]:
    url = f"https://pypi.org/pypi/{PROJECT}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)["releases"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise


def cmd_preflight(args: argparse.Namespace) -> int:
    version = version_from_tag(args.tag) if args.tag else None
    releases = fetch_pypi_releases()

    sizes = {v: sum(f["size"] for f in files) for v, files in releases.items()}
    used = sum(sizes.values())
    dev = {v: s for v, s in sizes.items() if not re.fullmatch(r"\d+\.\d+\.\d+", v)}

    final = [(v, s) for v, s in sizes.items() if v not in dev and s > 0]
    final.sort(key=lambda vs: tuple(int(p) for p in vs[0].split(".")))
    estimate = final[-1][1] if final else DEFAULT_RELEASE_BYTES

    _summary(f"### PyPI storage for `{PROJECT}`")
    _summary(f"- Used: **{_gb(used)}** of {_gb(PROJECT_LIMIT_BYTES)} "
             f"({used / PROJECT_LIMIT_BYTES:.0%}) across {len(sizes)} versions")
    _summary(f"- Non-release (dev/rc/post) versions: {len(dev)}, {_gb(sum(dev.values()))}")
    _summary(f"- Estimated size of one release: {_mb(estimate)}")
    room = int(PROJECT_LIMIT_BYTES * MAX_USAGE_FRACTION) - used
    _summary(f"- Room for about **{max(room, 0) // estimate}** more releases "
             f"before the {MAX_USAGE_FRACTION:.0%} safety line")

    if dev:
        biggest = sorted(dev.items(), key=lambda vs: -vs[1])[:10]
        _summary("- Largest non-release versions (delete them on pypi.org to free space): "
                 + ", ".join(f"`{v}`" for v, _ in biggest))

    if version is not None:
        if version in releases and releases[version]:
            _summary(f"- `{version}` already has files on PyPI; only missing files "
                     "will be uploaded (skip-existing).")
        elif used + estimate > PROJECT_LIMIT_BYTES * MAX_USAGE_FRACTION:
            _summary(
                f"\nERROR: publishing {version} (~{_mb(estimate)}) would take PyPI usage "
                f"to {_gb(used + estimate)}, above the {MAX_USAGE_FRACTION:.0%} safety line. "
                "Delete old dev versions at "
                f"https://pypi.org/manage/project/{PROJECT}/releases/ and re-run."
            )
            return 1

    if used > PROJECT_LIMIT_BYTES * WARN_USAGE_FRACTION:
        _summary(f"\nWARNING: PyPI usage is above {WARN_USAGE_FRACTION:.0%}; "
                 "consider deleting old versions soon.")
    return 0


def cmd_verify_dist(args: argparse.Namespace) -> int:
    version = version_from_tag(args.tag)
    files = sorted(p for p in Path(args.dist).iterdir() if p.is_file())
    if not files:
        print(f"ERROR: no files found in {args.dist}")
        return 1

    bad = []
    for p in files:
        # Wheels: name-version-... ; sdist: name-version.tar.gz
        m = re.match(rf"^{PROJECT}-([^-]+?)(?:-|\.tar\.gz$)", p.name)
        file_version = m.group(1) if m else None
        if file_version != version:
            bad.append(f"{p.name} (version {file_version!r}, expected {version!r})")

    total = sum(p.stat().st_size for p in files)
    _summary(f"### Upload for `{PROJECT}` {version}")
    _summary(f"- {len(files)} files, {_mb(total)} total")

    if bad:
        _summary("\nERROR: refusing to upload files that are not exactly version "
                 f"{version} (dev/local builds must never reach PyPI):")
        for line in bad:
            _summary(f"  - {line}")
        return 1
    if total > MAX_RELEASE_BYTES:
        _summary(f"\nERROR: upload is {_mb(total)}, above the {_mb(MAX_RELEASE_BYTES)} "
                 "per-release budget. Check for debug or duplicated wheels, or raise "
                 "MAX_RELEASE_BYTES in tools/check_pypi_release.py on purpose.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="check tag format and PyPI quota")
    p.add_argument("--tag", help="release tag, e.g. v0.7.2 (omit to only report usage)")
    p.set_defaults(func=cmd_preflight)

    v = sub.add_parser("verify-dist", help="check built files before upload")
    v.add_argument("dist", help="directory containing the wheels and sdist")
    v.add_argument("--tag", required=True)
    v.set_defaults(func=cmd_verify_dist)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
