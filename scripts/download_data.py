#!/usr/bin/env python3
"""Download and verify the FlyWire datasets used by this model.

The original upstream repository tracks the data files alongside the model.
This repository keeps those large files out of Git and fetches the exact
upstream revision on demand instead.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

UPSTREAM_REPOSITORY = "philshiu/Drosophila_brain_model"
UPSTREAM_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_REPOSITORY}.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DataFile:
    """A raw dataset file pinned to the imported upstream revision."""

    filename: str
    size: int
    sha256: str

    @property
    def source_url(self) -> str:
        return (
            f"https://github.com/{UPSTREAM_REPOSITORY}/blob/"
            f"{UPSTREAM_COMMIT}/{self.filename}"
        )


DATASETS: dict[str, tuple[DataFile, ...]] = {
    "v630": (
        DataFile(
            "2023_03_23_completeness_630_final.csv",
            3_057_611,
            "e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb",
        ),
        DataFile(
            "2023_03_23_connectivity_630_final.parquet",
            86_630_944,
            "94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3",
        ),
    ),
    "v783": (
        DataFile(
            "Completeness_783.csv",
            3_327_347,
            "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        ),
        DataFile(
            "Connectivity_783.parquet",
            100_804_642,
            "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        ),
    ),
}


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_error(path: Path, data_file: DataFile) -> str | None:
    """Return a human-readable validation failure, or None when valid."""
    if not path.is_file():
        return "missing"
    actual_size = path.stat().st_size
    if actual_size != data_file.size:
        return f"has {actual_size:,} bytes; expected {data_file.size:,}"
    actual_digest = file_digest(path)
    if actual_digest != data_file.sha256:
        return "SHA-256 does not match the pinned upstream file"
    return None


def format_size(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MiB"


def run_git(arguments: list[str], cwd: Path | None = None) -> None:
    """Run Git without ever asking the user for credentials interactively."""
    command = ["git", *arguments]
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required to download the connectome data.") from error

    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(f"Git command failed: {' '.join(command)}{suffix}")


def install_file(data_file: DataFile, destination: Path, source: Path) -> None:
    """Verify a sparse-checked-out file and atomically put it in place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    try:
        shutil.copyfile(source, partial)
        problem = validation_error(partial, data_file)
        if problem:
            raise RuntimeError(f"copied {data_file.filename}, but it {problem}")
        os.replace(partial, destination)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"could not install {destination}: {error}") from error
    except RuntimeError:
        partial.unlink(missing_ok=True)
        raise


def fetch_and_install(data_files: Iterable[DataFile], directory: Path) -> None:
    """Fetch the named files sparsely, verify them, and atomically install them."""
    requested = tuple(data_files)
    if not requested:
        return

    with tempfile.TemporaryDirectory(prefix="drosophila-brain-data-") as temp:
        repository = Path(temp) / "upstream"
        run_git(["init", "--quiet", str(repository)])
        run_git(["remote", "add", "origin", UPSTREAM_URL], repository)
        run_git(
            [
                "-c",
                "protocol.version=2",
                "fetch",
                "--quiet",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                UPSTREAM_COMMIT,
            ],
            repository,
        )
        run_git(["sparse-checkout", "init", "--no-cone"], repository)
        run_git(
            ["sparse-checkout", "set", "--no-cone", "--", *(f.filename for f in requested)],
            repository,
        )
        run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], repository)

        revision_result = subprocess.run(
            ["git", "rev-parse", "FETCH_HEAD"],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        fetched_revision = revision_result.stdout.strip()
        if fetched_revision != UPSTREAM_COMMIT:
            raise RuntimeError(
                "Git fetched an unexpected upstream revision: "
                f"{fetched_revision} (expected {UPSTREAM_COMMIT})"
            )

        for data_file in requested:
            source = repository / data_file.filename
            problem = validation_error(source, data_file)
            if problem:
                raise RuntimeError(
                    f"the upstream checkout's {data_file.filename} {problem}"
                )
            print(f"Installing {data_file.filename} ({format_size(data_file.size)})")
            install_file(data_file, directory / data_file.filename, source)
            print(f"Verified {data_file.filename}")


def dataset_files(dataset: str) -> Iterable[DataFile]:
    return DATASETS[dataset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the FlyWire connectome data pinned to the "
            "imported Drosophila brain model."
        )
    )
    parser.add_argument(
        "--dataset",
        "--version",
        choices=sorted(DATASETS),
        default="v630",
        help="FlyWire dataset to install (default: v630, used by the original notebooks)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=PROJECT_ROOT,
        help="directory where the model expects the data files (default: project root)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only validate files already present; do not download anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a missing, incomplete, or mismatched local file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the pinned files and source URLs without changing anything",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination_directory = args.directory.expanduser().resolve()
    failures: list[str] = []
    needed: list[DataFile] = []

    print(
        f"Dataset {args.dataset} from {UPSTREAM_REPOSITORY}@{UPSTREAM_COMMIT[:12]}"
    )
    for data_file in dataset_files(args.dataset):
        destination = destination_directory / data_file.filename
        problem = validation_error(destination, data_file)

        if args.dry_run:
            status = "ready" if problem is None else problem
            print(f"{data_file.filename} ({format_size(data_file.size)}): {status}")
            print(f"  {data_file.source_url}")
            continue

        if problem is None:
            print(f"Verified {destination.name}")
            continue

        if args.verify:
            failures.append(f"{destination}: {problem}")
            continue

        if destination.exists() and not args.force:
            failures.append(
                f"{destination}: {problem}. Re-run with --force to replace it."
            )
            continue

        needed.append(data_file)

    if failures:
        print("\nData setup did not complete:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    if args.dry_run:
        return 0

    if needed:
        names = ", ".join(data_file.filename for data_file in needed)
        print(f"Fetching from GitHub: {names}")
        try:
            fetch_and_install(needed, destination_directory)
        except RuntimeError as error:
            print(f"\nData setup did not complete:\n- {error}", file=sys.stderr)
            return 1

    print("\nData setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
