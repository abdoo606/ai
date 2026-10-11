#!/usr/bin/env python3
"""Install a keyless local Jarvis model profile once.

One strong multilingual reasoning model is shared in memory by four logical
roles (reasoner, critic, Arabic writer, English writer). A smaller independent
model handles intent. This is much stronger and more memory-efficient than
loading separate duplicate weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROFILES = {
    "compact": {
        "intent": "Qwen/Qwen3-0.6B",
        "reasoning": "Qwen/Qwen3-4B",
        "guidance": "about 12 GB system RAM; entry-level local reasoning",
    },
    "balanced": {
        "intent": "Qwen/Qwen3-0.6B",
        "reasoning": "Qwen/Qwen3-8B",
        "guidance": "about 20 GB system RAM; stronger everyday reasoning",
    },
    "powerful": {
        "intent": "Qwen/Qwen3-1.7B",
        "reasoning": "Qwen/Qwen3-14B",
        "guidance": "about 36 GB system RAM; recommended high-quality profile",
    },
    "max": {
        "intent": "Qwen/Qwen3-1.7B",
        "reasoning": "Qwen/Qwen3-32B",
        "guidance": "about 80 GB system RAM or suitable accelerator; strong workstation profile",
    },
    "frontier": {
        "intent": "Qwen/Qwen3-1.7B",
        "reasoning": "Qwen/Qwen3-235B-A22B",
        "guidance": "roughly 500 GB unquantized memory and multi-GPU/server hardware",
    },
}
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Jarvis's local intent and reasoning brains")
    parser.add_argument("--profile", choices=PROFILES, default="powerful",
                        help="hardware/quality profile (default: powerful)")
    parser.add_argument("--only", choices=["all", "intent", "reasoning"], default="all")
    parser.add_argument("--directory", type=Path, default=ROOT / "models",
                        help="model destination (default: jarvis/models)")
    parser.add_argument("--reasoning", help="override reasoning model repository ID")
    parser.add_argument("--intent", help="override intent model repository ID")
    parser.add_argument("--reasoning-revision", help="immutable commit SHA (resolved automatically if omitted)")
    parser.add_argument("--intent-revision", help="immutable commit SHA (resolved automatically if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="show downloads without fetching")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    configured = {
        "intent": args.intent or profile["intent"],
        "reasoning": args.reasoning or profile["reasoning"],
    }
    roles = list(configured) if args.only == "all" else [args.only]
    print(f"Profile: {args.profile} — {profile['guidance']}")
    requested_revisions = {"intent": args.intent_revision, "reasoning": args.reasoning_revision}
    for role in roles:
        revision = requested_revisions[role] or "<resolve-current-commit>"
        print(f"{role}: {configured[role]}@{revision} -> {args.directory / role}")
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit("Install first: pip install -r jarvis/requirements-huggingface.txt") from error

    api = HfApi()
    requested_revisions = {"intent": args.intent_revision, "reasoning": args.reasoning_revision}
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / "manifest.json"
    manifest = {"installed_at": time.time(), "profile": args.profile, "models": {}}
    if manifest_path.exists():
        manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["installed_at"] = time.time()
        manifest["profile"] = args.profile
    for role in roles:
        repo_id = configured[role]
        destination = args.directory / role
        revision = requested_revisions[role] or api.model_info(repo_id).sha
        if not revision:
            raise RuntimeError(f"could not resolve an immutable revision for {repo_id}")
        print(f"\nInstalling {role} brain: {repo_id}@{revision}")
        resolved = snapshot_download(repo_id=repo_id, revision=revision, local_dir=destination)
        manifest.setdefault("models", {})[role] = {
            "repo_id": repo_id, "revision": revision,
            "path": str(destination), "resolved": resolved,
        }
        print(f"Ready: {role}")
    manifest["logical_roles"] = {
        "intent": str(args.directory / "intent"),
        "reasoner": str(args.directory / "reasoning"),
        "critic": str(args.directory / "reasoning"),
        "arabic_writer": str(args.directory / "reasoning"),
        "english_writer": str(args.directory / "reasoning"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nReady. Start strong mode with: JARVIS_BACKEND=powerful python -m jarvis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
