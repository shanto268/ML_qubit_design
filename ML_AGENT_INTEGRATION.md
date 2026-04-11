# SQuADDS ML Agent Integration Guide

This repo now includes two deployable surfaces for AI agents and automation:

1. A Hugging Face Docker Space with a stable HTTP API.
2. Per-model Hugging Face model repos with machine-readable inference manifests.

## Current Live Endpoints

- Space repo: [https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api](https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api)
- Space host: [https://squadds-squadds-ml-inference-api.hf.space](https://squadds-squadds-ml-inference-api.hf.space)
- First published model repo: [https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse](https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse)

## Primary Tool Surface

Agents should prefer the Space when they want a single HTTP tool with model
discovery and a consistent request contract.

### Endpoints

- `GET /health`
- `GET /models`
- `POST /predict`

### Human Quickstart

```bash
curl https://squadds-squadds-ml-inference-api.hf.space/models
```

```bash
curl -X POST \
  https://squadds-squadds-ml-inference-api.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"transmon_cross_hamiltonian_inverse","inputs":{"qubit_frequency_GHz":4.85,"anharmonicity_MHz":-205.0}}'
```

### Request Contract

```json
{
  "model_id": "transmon_cross_hamiltonian_inverse",
  "inputs": {
    "qubit_frequency_GHz": 4.85,
    "anharmonicity_MHz": -205.0
  },
  "options": {
    "include_scaled_outputs": false
  }
}
```

### Response Shape

```json
{
  "model_id": "transmon_cross_hamiltonian_inverse",
  "display_name": "TransmonCross Hamiltonian to Geometry",
  "predictions": [
    {
      "design_options.connection_pads.readout.claw_length": 0.00023,
      "design_options.connection_pads.readout.ground_spacing": 0.00001,
      "design_options.cross_length": 0.00019
    }
  ],
  "metadata": {
    "input_order": [
      "qubit_frequency_GHz",
      "anharmonicity_MHz"
    ],
    "output_order": [
      "design_options.connection_pads.readout.claw_length",
      "design_options.connection_pads.readout.ground_spacing",
      "design_options.cross_length"
    ],
    "input_units": {
      "qubit_frequency_GHz": "GHz",
      "anharmonicity_MHz": "MHz"
    },
    "output_units": {
      "design_options.connection_pads.readout.claw_length": "m",
      "design_options.connection_pads.readout.ground_spacing": "m",
      "design_options.cross_length": "m"
    },
    "num_predictions": 1
  }
}
```

## Agent Workflow Recommendation

1. Call `GET /models` first to discover currently bundled models.
2. Select a model whose `status` is `ready`.
3. Build a `POST /predict` request using exactly the input keys listed by that model.
4. Use the returned geometry values directly or feed them into downstream
   SQuADDS / Qiskit Metal / validation workflows.

### Current Live Example

Request:

```json
{
  "model_id": "transmon_cross_hamiltonian_inverse",
  "inputs": {
    "qubit_frequency_GHz": 4.85,
    "anharmonicity_MHz": -205.0
  }
}
```

Observed response:

```json
{
  "model_id": "transmon_cross_hamiltonian_inverse",
  "display_name": "TransmonCross Hamiltonian to Geometry",
  "predictions": [
    {
      "design_options.connection_pads.readout.claw_length": 0.00011072495544794947,
      "design_options.connection_pads.readout.ground_spacing": 4.571595582092414e-06,
      "design_options.cross_length": 0.0002005973074119538
    }
  ],
  "metadata": {
    "input_order": [
      "qubit_frequency_GHz",
      "anharmonicity_MHz"
    ],
    "output_order": [
      "design_options.connection_pads.readout.claw_length",
      "design_options.connection_pads.readout.ground_spacing",
      "design_options.cross_length"
    ],
    "input_units": {
      "qubit_frequency_GHz": "GHz",
      "anharmonicity_MHz": "MHz"
    },
    "output_units": {
      "design_options.connection_pads.readout.claw_length": "m",
      "design_options.connection_pads.readout.ground_spacing": "m",
      "design_options.cross_length": "m"
    },
    "num_predictions": 1
  }
}
```

## Model Repos

Each Hugging Face model repo built from this project includes:

- the trained Keras checkpoint
- scaler files
- input/output name files
- `inference_manifest.json`

Agents can use the model repo manifests to understand the contract even when
they are not calling the Space directly.

## Local Build + Publish

### Build the Space bundle

```bash
python3 scripts/build_hf_space_bundle.py
```

### Publish the Space

```bash
python3 scripts/publish_hf_space.py
```

### Build the model repos

```bash
python3 scripts/build_hf_model_repos.py
```

### Publish the model repos

```bash
python3 scripts/publish_hf_model_repos.py
```

## Current Artifact Reality

The deployment tooling is manifest-driven and supports all four model families
defined in `hf_space_api/source_models.json`, but it only bundles model folders
whose saved checkpoints are actually present locally.

At the time this integration was added, the immediately deployable artifact set
was the TransmonCross Hamiltonian inverse model. The remaining model families
will be auto-included once their checkpoint and scaler directories are added.
