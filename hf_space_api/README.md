# SQuADDS ML Hugging Face Space

This directory contains the source for a Docker Space that serves the trained
`ML_qubit_design` models behind a documented HTTP API.

Use the bundle builder to create a deployable Space repo with only the code and
artifacts needed for inference:

```bash
python3 scripts/build_hf_space_bundle.py
```

The resulting bundle is written to `dist/squadds-ml-inference-api` and can be
published to Hugging Face with:

```bash
python3 scripts/publish_hf_space.py
```

The Space exposes:

- `GET /health`
- `GET /models`
- `POST /predict`

Live endpoints:

- Space repo: `https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api`
- Space host: `https://squadds-squadds-ml-inference-api.hf.space`

Sample prediction:

```bash
curl -X POST \
  https://squadds-squadds-ml-inference-api.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"transmon_cross_hamiltonian_inverse","inputs":{"qubit_frequency_GHz":4.85,"anharmonicity_MHz":-205.0}}'
```

The builder currently packages every model whose required artifacts are present
in this repo and skips incomplete model folders with a clear reason.
