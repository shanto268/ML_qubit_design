#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "hf_space_api"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "squadds-ml-inference-api"


def load_source_config() -> Dict[str, Any]:
    return json.loads((TEMPLATE_ROOT / "source_models.json").read_text())


def copy_template(output_dir: Path) -> None:
    ignore_names = shutil.ignore_patterns("__pycache__", "*.pyc", "source_models.json")
    shutil.copytree(TEMPLATE_ROOT, output_dir, ignore=ignore_names)


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


def copy_model_artifacts(entry: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    source_dir = REPO_ROOT / entry["source_dir"]
    artifact_dir = output_dir / "artifacts" / entry["id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_target = artifact_dir / entry["resolved_model_path"]
    model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / entry["resolved_model_path"], model_target)

    input_target = artifact_dir / entry["input_names_path"]
    input_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / entry["input_names_path"], input_target)

    output_target = artifact_dir / entry["output_names_path"]
    output_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / entry["output_names_path"], output_target)

    for extra_path in entry.get("copy_paths", []):
        source_path = source_dir / extra_path
        if not source_path.exists():
            continue
        target_path = artifact_dir / extra_path
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    return {
        "id": entry["id"],
        "display_name": entry["display_name"],
        "description": entry["description"],
        "artifact_dir": str(Path("artifacts") / entry["id"]),
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
        "status": "ready",
        "status_detail": "",
        "tags": entry.get("tags", []),
        "prediction_output_index": entry.get("prediction_output_index", 0),
    }


def make_skipped_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry["id"],
        "display_name": entry["display_name"],
        "description": entry["description"],
        "artifact_dir": "",
        "model_path": "",
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
        "status": entry["status"],
        "status_detail": entry.get("status_detail", ""),
        "tags": entry.get("tags", []),
        "prediction_output_index": entry.get("prediction_output_index", 0),
    }


def render_space_readme(space: Dict[str, Any], ready: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> str:
    lines = [
        "---",
        f"title: {space['title']}",
        f"sdk: {space['sdk']}",
        f"app_port: {space['app_port']}",
        f"license: {space['license']}",
        "---",
        "",
        f"# {space['title']}",
        "",
        "Auto-generated deployment bundle for serving ML_qubit_design models with a FastAPI app.",
        "",
        "## Live URLs",
        "",
        f"- Space repo: {space.get('repo_url', '')}",
        f"- Space host: {space.get('host', '')}",
        "",
        "## Endpoints",
        "",
        "- `GET /health`",
        "- `GET /models`",
        "- `POST /predict`",
        "",
        "## Quickstart",
        "",
        "List models:",
        "",
        "```bash",
        f"curl {space.get('host', '')}/models",
        "```",
        "",
        "## Included Models",
        "",
    ]

    if ready:
        example = ready[0]
        sample_request = example.get("sample_request")
        sample_response = example.get("sample_response")
        if sample_request and sample_response:
            lines.extend(
                [
                    "Run a sample prediction:",
                    "",
                    "```bash",
                    "curl -X POST \\",
                    f"  {space.get('host', '')}/predict \\",
                    "  -H 'Content-Type: application/json' \\",
                    f"  -d '{json.dumps(sample_request)}'",
                    "```",
                    "",
                    "Sample response:",
                    "",
                    "```json",
                    json.dumps(sample_response, indent=2),
                    "```",
                    "",
                ]
            )
        for model in ready:
            lines.append(f"- `{model['id']}`: {model['description']}")
    else:
        lines.append("- No models were bundled.")

    lines.extend(["", "## Skipped Models", ""])
    if skipped:
        for model in skipped:
            lines.append(f"- `{model['id']}`: {model['status_detail'] or model['status']}")
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def build_bundle(output_dir: Path) -> Dict[str, Any]:
    config = load_source_config()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    copy_template(output_dir)

    resolved_entries = [resolve_model_entry(item) for item in config["models"]]
    ready_entries = [item for item in resolved_entries if item["status"] == "ready"]
    skipped_entries = [item for item in resolved_entries if item["status"] != "ready"]

    ready_manifest = [copy_model_artifacts(item, output_dir) for item in ready_entries]
    skipped_manifest = [make_skipped_entry(item) for item in skipped_entries]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "space": config["space"],
        "models": ready_manifest + skipped_manifest,
    }

    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        render_space_readme(config["space"], ready_manifest, skipped_manifest)
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deployable Hugging Face Docker Space bundle for ML_qubit_design."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the generated bundle.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    manifest = build_bundle(output_dir)

    ready = [item for item in manifest["models"] if item["status"] == "ready"]
    skipped = [item for item in manifest["models"] if item["status"] != "ready"]
    print(f"Bundle created at {output_dir}")
    print(f"Ready models: {[item['id'] for item in ready]}")
    print(f"Skipped models: {[item['id'] for item in skipped]}")


if __name__ == "__main__":
    main()
