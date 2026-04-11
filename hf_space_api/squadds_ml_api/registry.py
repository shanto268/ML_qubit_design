from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
from keras.models import load_model


class BundleConfigError(RuntimeError):
    """Raised when the deployed bundle is missing required files or config."""


class RequestValidationError(ValueError):
    """Raised when an inference request does not match the model contract."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    description: str
    artifact_dir: str
    model_path: str
    input_names_path: str
    input_names_format: str
    output_names_path: str
    output_names_format: str
    x_scaler_pattern: str | None = None
    y_scaler_pattern: str | None = None
    input_units: Dict[str, str] = field(default_factory=dict)
    output_units: Dict[str, str] = field(default_factory=dict)
    sample_request: Dict[str, Any] | None = None
    sample_response: Dict[str, Any] | None = None
    status: str = "ready"
    status_detail: str = ""
    tags: List[str] = field(default_factory=list)
    prediction_output_index: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelSpec":
        return cls(**payload)


def _read_names(path: Path, fmt: str) -> List[str]:
    if fmt == "text_lines":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if fmt == "csv_header":
        first_line = path.read_text().splitlines()[0]
        return [item.strip() for item in first_line.split(",") if item.strip()]
    if fmt == "npy":
        values = np.load(path, allow_pickle=True)
        return [str(item) for item in values.tolist()]
    raise BundleConfigError(f"Unsupported name file format: {fmt}")


class EndpointModel:
    def __init__(self, bundle_root: Path, spec: ModelSpec):
        self.bundle_root = bundle_root
        self.spec = spec
        self.artifact_root = bundle_root / spec.artifact_dir

        if not self.artifact_root.exists():
            raise BundleConfigError(
                f"Artifact directory for model '{spec.id}' is missing: {self.artifact_root}"
            )

        self.input_names = _read_names(
            self.artifact_root / spec.input_names_path, spec.input_names_format
        )
        self.output_names = _read_names(
            self.artifact_root / spec.output_names_path, spec.output_names_format
        )

        model_file = self.artifact_root / spec.model_path
        if not model_file.exists():
            raise BundleConfigError(
                f"Model file for '{spec.id}' is missing: {model_file}"
            )
        self.model = load_model(model_file, compile=False)

    def _load_scaler(self, pattern: str, name: str):
        scaler_path = self.artifact_root / pattern.format(name=name)
        if not scaler_path.exists():
            raise BundleConfigError(
                f"Required scaler for '{self.spec.id}' is missing: {scaler_path}"
            )
        return joblib.load(scaler_path)

    def _normalize_rows(self, inputs: Dict[str, float] | List[Dict[str, float]]) -> List[Dict[str, float]]:
        rows = inputs if isinstance(inputs, list) else [inputs]
        normalized: List[Dict[str, float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RequestValidationError(
                    f"Each input row must be an object, got {type(row).__name__} at index {index}."
                )

            missing = [name for name in self.input_names if name not in row]
            extras = [name for name in row if name not in self.input_names]
            if missing or extras:
                raise RequestValidationError(
                    f"Model '{self.spec.id}' expects inputs {self.input_names}. "
                    f"Missing: {missing or 'none'}. Extra: {extras or 'none'}."
                )

            normalized.append({name: float(row[name]) for name in self.input_names})
        return normalized

    def _scale_inputs(self, rows: List[Dict[str, float]]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(self.input_names)), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for col_idx, name in enumerate(self.input_names):
                value = row[name]
                if self.spec.x_scaler_pattern:
                    scaler = self._load_scaler(self.spec.x_scaler_pattern, name)
                    scaled = scaler.transform([[value]])[0][0]
                else:
                    scaled = value
                matrix[row_idx, col_idx] = float(scaled)
        return matrix

    def _unscale_outputs(self, scaled_outputs: np.ndarray) -> np.ndarray:
        matrix = np.asarray(scaled_outputs, dtype=np.float32).copy()
        if not self.spec.y_scaler_pattern:
            return matrix

        for col_idx, name in enumerate(self.output_names):
            scaler = self._load_scaler(self.spec.y_scaler_pattern, name)
            column = matrix[:, col_idx].reshape(-1, 1)
            matrix[:, col_idx] = scaler.inverse_transform(column).reshape(-1)
        return matrix

    def predict(
        self,
        inputs: Dict[str, float] | List[Dict[str, float]],
        include_scaled_outputs: bool = False,
    ) -> Dict[str, Any]:
        rows = self._normalize_rows(inputs)
        scaled_inputs = self._scale_inputs(rows)
        raw_predictions = self.model.predict(scaled_inputs, verbose=0)

        if isinstance(raw_predictions, list):
            scaled_outputs = np.asarray(raw_predictions[self.spec.prediction_output_index])
        else:
            scaled_outputs = np.asarray(raw_predictions)

        unscaled_outputs = self._unscale_outputs(scaled_outputs)

        predictions = [
            {
                output_name: float(unscaled_outputs[row_idx, col_idx])
                for col_idx, output_name in enumerate(self.output_names)
            }
            for row_idx in range(unscaled_outputs.shape[0])
        ]

        response: Dict[str, Any] = {
            "model_id": self.spec.id,
            "display_name": self.spec.display_name,
            "predictions": predictions,
            "metadata": {
                "input_order": self.input_names,
                "output_order": self.output_names,
                "input_units": self.spec.input_units,
                "output_units": self.spec.output_units,
                "num_predictions": len(predictions),
            },
        }
        if include_scaled_outputs:
            response["scaled_predictions"] = [
                {
                    output_name: float(scaled_outputs[row_idx, col_idx])
                    for col_idx, output_name in enumerate(self.output_names)
                }
                for row_idx in range(scaled_outputs.shape[0])
            ]
        return response


class ModelRegistry:
    def __init__(self, bundle_root: Path):
        self.bundle_root = Path(bundle_root)
        manifest_path = self.bundle_root / "deployment_manifest.json"
        if not manifest_path.exists():
            raise BundleConfigError(
                f"Deployment manifest is missing from bundle: {manifest_path}"
            )

        payload = json.loads(manifest_path.read_text())
        self.bundle_info = payload.get("space", {})
        self.specs = [ModelSpec.from_dict(item) for item in payload.get("models", [])]
        self._models: Dict[str, EndpointModel] = {}

    def available_model_ids(self) -> List[str]:
        return [spec.id for spec in self.specs if spec.status == "ready"]

    def describe_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": spec.id,
                "display_name": spec.display_name,
                "description": spec.description,
                "status": spec.status,
                "status_detail": spec.status_detail,
                "input_units": spec.input_units,
                "output_units": spec.output_units,
                "tags": spec.tags,
            }
            for spec in self.specs
        ]

    def _get_model(self, model_id: str) -> EndpointModel:
        spec = next((item for item in self.specs if item.id == model_id), None)
        if spec is None:
            raise RequestValidationError(
                f"Unknown model_id '{model_id}'. Available models: {self.available_model_ids()}."
            )
        if spec.status != "ready":
            raise RequestValidationError(
                f"Model '{model_id}' is not deployable in this bundle: {spec.status_detail or spec.status}."
            )
        if model_id not in self._models:
            self._models[model_id] = EndpointModel(self.bundle_root, spec)
        return self._models[model_id]

    def predict(
        self,
        model_id: str,
        inputs: Dict[str, float] | List[Dict[str, float]],
        include_scaled_outputs: bool = False,
    ) -> Dict[str, Any]:
        model = self._get_model(model_id)
        return model.predict(inputs=inputs, include_scaled_outputs=include_scaled_outputs)
