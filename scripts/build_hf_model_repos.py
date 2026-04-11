#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "hf_space_api"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "hf_model_repos"


def load_source_config() -> Dict[str, Any]:
    return json.loads((TEMPLATE_ROOT / "source_models.json").read_text())


def resolve_model_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    source_dir = REPO_ROOT / config["source_dir"]
    resolved = dict(config)
    resolved["status"] = "ready"
    resolved["status_detail"] = ""

    if not source_dir.exists():
        resolved["status"] = "missing_source_dir"
        resolved["status_detail"] = f"Missing source directory: {source_dir}"
        return resolved

    resolved_model = None
    for candidate in config.get("model_candidates", []):
        if (source_dir / candidate).exists():
            resolved_model = candidate
            break
    if resolved_model is None:
        resolved["status"] = "missing_model_artifact"
        resolved["status_detail"] = (
            "No model checkpoint found. Expected one of: "
            + ", ".join(config.get("model_candidates", []))
        )
        return resolved

    required_files = [
        config["input_names_path"],
        config["output_names_path"],
        resolved_model,
    ]
    missing = [item for item in required_files if not (source_dir / item).exists()]
    if missing:
        resolved["status"] = "missing_required_files"
        resolved["status_detail"] = "Missing required files: " + ", ".join(missing)
        return resolved

    resolved["resolved_model_path"] = resolved_model
    return resolved


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def render_model_card(entry: Dict[str, Any]) -> str:
    tags = entry.get("tags", []) + ["squadds", "keras", "tabular-regression"]
    unique_tags = []
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)

    lines = [
        "---",
        "library_name: keras",
        "license: mit",
        "tags:",
    ]
    for tag in unique_tags:
        lines.append(f"- {tag}")
    lines.extend(
        [
            "---",
            "",
            f"# {entry['display_name']}",
            "",
            entry["description"],
            "",
            "## Live Serving Surface",
            "",
            "- Space repo: https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api",
            "- Space host: https://squadds-squadds-ml-inference-api.hf.space",
            "",
            "## Inference Contract",
            "",
            "The deployed artifact uses the same request contract as the SQuADDS ML Space:",
            "",
            "```json",
            json.dumps(
                entry.get("sample_request")
                or {"model_id": entry["id"], "inputs": {"...": 0.0}},
                indent=2,
            ),
            "```",
            "",
            "## Sample Response",
            "",
            "```json",
            json.dumps(
                entry.get("sample_response") or {"predictions": [{"...": 0.0}]},
                indent=2,
            ),
            "```",
            "",
            "## Input and Output Fields",
            "",
            f"- Input units: `{json.dumps(entry.get('input_units', {}), sort_keys=True)}`",
            f"- Output units: `{json.dumps(entry.get('output_units', {}), sort_keys=True)}`",
            "",
            "## Included Files",
            "",
            "- `model/`: trained Keras checkpoint",
            "- `scalers/`: per-column input and output scalers when available",
            "- `X_names`: ordered input feature names",
            "- output-name file (`y_columns.npy` or csv header source)",
            "- `inference_manifest.json`: machine-readable contract for agents and clients",
            "",
            "## Suggested Use",
            "",
            "Use this repo as a durable artifact source and use the SQuADDS ML Space when you want",
            "a stable HTTP tool surface for agents or applications.",
            "",
        ]
    )
    return "\n".join(lines)


def build_model_repo(entry: Dict[str, Any], output_root: Path) -> Path:
    source_dir = REPO_ROOT / entry["source_dir"]
    repo_name = entry["hf_model_repo_id"].split("/", 1)[1]
    repo_dir = output_root / repo_name
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    copy_path(source_dir / entry["resolved_model_path"], repo_dir / entry["resolved_model_path"])
    copy_path(source_dir / entry["input_names_path"], repo_dir / entry["input_names_path"])
    copy_path(source_dir / entry["output_names_path"], repo_dir / entry["output_names_path"])
    for extra_path in entry.get("copy_paths", []):
        source_path = source_dir / extra_path
        if source_path.exists():
            copy_path(source_path, repo_dir / extra_path)

    manifest = {
        "model_id": entry["id"],
        "display_name": entry["display_name"],
        "description": entry["description"],
        "hf_model_repo_id": entry["hf_model_repo_id"],
        "model_path": entry["resolved_model_path"],
        "input_names_path": entry["input_names_path"],
        "input_names_format": entry["input_names_format"],
        "output_names_path": entry["output_names_path"],
        "output_names_format": entry["output_names_format"],
        "x_scaler_pattern": entry.get("x_scaler_pattern"),
        "y_scaler_pattern": entry.get("y_scaler_pattern"),
        "input_units": entry.get("input_units", {}),
        "output_units": entry.get("output_units", {}),
        "sample_request": entry.get("sample_request"),
        "sample_response": entry.get("sample_response"),
        "tags": entry.get("tags", []),
    }
    (repo_dir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (repo_dir / "README.md").write_text(render_model_card(entry))
    return repo_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-model Hugging Face model repo directories from ML_qubit_design artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where model repo folders should be created.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_config()
    resolved = [resolve_model_entry(item) for item in source["models"]]
    ready = [item for item in resolved if item["status"] == "ready"]
    skipped = [item for item in resolved if item["status"] != "ready"]

    built_dirs = [build_model_repo(item, output_dir) for item in ready]
    print(f"Built model repos under {output_dir}")
    print(f"Ready model repos: {[path.name for path in built_dirs]}")
    print(f"Skipped model repos: {[item['id'] for item in skipped]}")


if __name__ == "__main__":
    main()
