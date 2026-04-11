#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "dist" / "hf_model_repos"
DEFAULT_ENV_FILE = Path("/Users/shanto/LFL/SQuADDS/SQuADDS/.env")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.replace("export ", "").strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def resolve_token(env_file: Path | None) -> str:
    token = os.getenv("HUGGINGFACE_API_KEY")
    if token:
        return token
    if env_file and env_file.exists():
        token = parse_env_file(env_file).get("HUGGINGFACE_API_KEY")
        if token:
            return token
    raise RuntimeError(
        "HUGGINGFACE_API_KEY is not available. Export it or pass a valid --env-file."
    )


def ensure_model_repo_exists(token: str, repo_id: str) -> None:
    org, name = repo_id.split("/", 1)
    payload = {
        "name": name,
        "organization": org,
        "private": False
    }
    request = urllib.request.Request(
        "https://huggingface.co/api/repos/create",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(f"Created model repo {repo_id}: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            print(f"Model repo {repo_id} already exists.")
            return
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to create model repo {repo_id}: {exc.code} {body}") from exc


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def publish_repo(repo_dir: Path, repo_id: str, token: str) -> None:
    remote_url = f"https://__token__:{token}@huggingface.co/{repo_id}"
    with tempfile.TemporaryDirectory(prefix="hf-model-publish-") as tmp:
        clone_dir = Path(tmp) / "repo"
        run(["git", "clone", remote_url, str(clone_dir)])

        for item in clone_dir.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        for item in repo_dir.iterdir():
            destination = clone_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        run(["git", "add", "."], cwd=clone_dir)
        diff = subprocess.run(
            ["git", "status", "--short"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        if not diff.stdout.strip():
            print(f"No changes to publish for {repo_id}.")
            return

        run(["git", "config", "user.name", "Codex"], cwd=clone_dir)
        run(["git", "config", "user.email", "codex@openai.com"], cwd=clone_dir)
        run(["git", "commit", "-m", "Update SQuADDS ML model artifacts"], cwd=clone_dir)
        run(["git", "push", "origin", "main"], cwd=clone_dir)
        print(f"Published {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish built Hugging Face model repos for the SQuADDS ML models."
    )
    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory produced by build_hf_model_repos.py.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Optional .env file containing HUGGINGFACE_API_KEY.",
    )
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    if not model_root.exists():
        raise RuntimeError(
            f"Model repo directory does not exist: {model_root}. Run build_hf_model_repos.py first."
        )

    env_file = Path(args.env_file).resolve() if args.env_file else None
    token = resolve_token(env_file)

    for repo_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        manifest_path = repo_dir / "inference_manifest.json"
        if not manifest_path.exists():
            print(f"Skipping {repo_dir.name}: missing inference_manifest.json")
            continue
        repo_id = json.loads(manifest_path.read_text())["hf_model_repo_id"]
        ensure_model_repo_exists(token, repo_id)
        publish_repo(repo_dir, repo_id, token)


if __name__ == "__main__":
    main()
