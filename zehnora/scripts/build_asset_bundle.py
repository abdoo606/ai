#!/usr/bin/env python3
"""Build a deterministic, source-only Zehnora buyer diligence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCLUDED_ROOT_FILES = {
    ".gitignore", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
}
INCLUDED_DIRECTORIES = {"zehnora", "sales", ".github"}
EXCLUDED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist",
    "data", "models", "sale_artifacts", "*.egg-info",
}
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _excluded(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_PARTS or part.endswith(".egg-info"):
            return True
    return path.suffix in {".pyc", ".pyo"}


def source_files(root: Path = PROJECT_ROOT) -> list[Path]:
    files = [root / name for name in INCLUDED_ROOT_FILES if (root / name).is_file()]
    for directory in INCLUDED_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file() and not _excluded(path.relative_to(root)))
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_bundle(output_dir: Path, version: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / f"zehnora-source-asset-{version}.zip"
    manifest_entries = []
    payloads = []
    for path in source_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        data = path.read_bytes()
        payloads.append((relative, data))
        manifest_entries.append({"path": relative, "size": len(data), "sha256": digest(data)})
    manifest = json.dumps({
        "product": "Zehnora",
        "version": version,
        "scope": "current source-only diligence bundle; no Git history, weights, runtime data, or credentials",
        "files": manifest_entries,
    }, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in payloads:
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo("ASSET_MANIFEST.json", FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest)

    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    checksum.write_text(f"{digest(bundle.read_bytes())}  {bundle.name}\n", encoding="ascii")
    return bundle, checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean Zehnora source-asset diligence bundle")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "sale_artifacts")
    parser.add_argument("--version", default="0.7.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle, checksum = build_bundle(args.output.resolve(), args.version)
    print(bundle)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
