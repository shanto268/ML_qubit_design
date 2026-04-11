from __future__ import annotations

from typing import Dict, List, Union

from pydantic import BaseModel, Field


class PredictionOptions(BaseModel):
    include_scaled_outputs: bool = Field(
        default=False,
        description="Include raw scaled model outputs alongside inverse-transformed values.",
    )


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="The deployed model identifier.")
    inputs: Union[Dict[str, float], List[Dict[str, float]]] = Field(
        ...,
        description="Either a single input object or a batch of input objects.",
    )
    options: PredictionOptions = Field(default_factory=PredictionOptions)
