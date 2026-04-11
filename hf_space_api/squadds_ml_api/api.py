from pathlib import Path

from fastapi import FastAPI, HTTPException

from .registry import BundleConfigError, ModelRegistry, RequestValidationError
from .schemas import PredictionRequest


def create_app(bundle_root: Path) -> FastAPI:
    registry = ModelRegistry(bundle_root)

    app = FastAPI(
        title="SQuADDS ML Inference API",
        version="0.1.0",
        description=(
            "HTTP API for running inference against ML models trained in "
            "ML_qubit_design and packaged for the SQuADDS Hugging Face Space."
        ),
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "SQuADDS ML Inference API",
            "docs": "/docs",
            "models_endpoint": "/models",
            "predict_endpoint": "/predict",
        }

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "available_models": registry.available_model_ids(),
            "bundle_root": str(bundle_root),
        }

    @app.get("/models")
    def list_models() -> dict:
        return {"models": registry.describe_models()}

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict:
        try:
            payload = registry.predict(
                model_id=request.model_id,
                inputs=request.inputs,
                include_scaled_outputs=request.options.include_scaled_outputs,
            )
        except RequestValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BundleConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected inference error for model '{request.model_id}': {exc}",
            ) from exc
        return payload

    return app
