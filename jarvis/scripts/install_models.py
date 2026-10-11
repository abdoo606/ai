#!/usr/bin/env python3
"""One-time installer for Jarvis's three public, local model brains.

No account or API key is used. Files are copied into jarvis/models and all
later inference can run with downloads disabled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELS = {
    # Small English instruction model (Apache-2.0).
    "english": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Multilingual instruction model selected for strong Arabic support (Apache-2.0).
    "arabic": "Qwen/Qwen2.5-1.5B-Instruct",
    # Smaller multilingual model used only for intent/tool classification (Apache-2.0).
    "intent": "Qwen/Qwen2.5-0.5B-Instruct",
}
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Jarvis's local English, Arabic and intent models")
    parser.add_argument("--only", choices=["all", *MODELS], default="all",
                        help="install all three brains or just one")
    parser.add_argument("--directory", type=Path, default=ROOT / "models",
                        help="model destination (default: jarvis/models)")
    parser.add_argument("--english", default=MODELS["english"], help="English model repository ID")
    parser.add_argument("--arabic", default=MODELS["arabic"], help="Arabic-capable model repository ID")
    parser.add_argument("--intent", default=MODELS["intent"], help="multilingual intent model repository ID")
    parser.add_argument("--dry-run", action="store_true", help="show downloads and destinations without fetching")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = MODELS if args.only == "all" else {args.only: MODELS[args.only]}
    configured = {"english": args.english, "arabic": args.arabic, "intent": args.intent}
    if args.dry_run:
        for role in selected:
            print(f"{role}: {configured[role]} -> {args.directory / role}")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Install first: pip install -r jarvis/requirements-huggingface.txt") from error

    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = {"installed_at": time.time(), "models": {}}
    for role in selected:
        repo_id = configured[role]
        destination = args.directory / role
        print(f"Installing {role} brain: {repo_id} -> {destination}")
        resolved = snapshot_download(repo_id=repo_id, local_dir=destination)
        manifest["models"][role] = {"repo_id": repo_id, "path": str(destination), "resolved": resolved}
        print(f"Ready: {role}")
    manifest_path = args.directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous.setdefault("models", {}).update(manifest["models"])
        previous["installed_at"] = manifest["installed_at"]
        manifest = previous
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nModels installed. Start with: JARVIS_BACKEND=ensemble python -m jarvis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
