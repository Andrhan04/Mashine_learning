# AGENTS.md

## Project Overview
- This repository is a compact educational RNN project built around the Jena climate dataset.
- The main workflow lives in `main.py`.
- Training data is stored locally in `jena_climate_2009_2016.csv`.
- Auxiliary raw material is stored in `materials/dataset.zip`.
- Model result plots are kept in `graphics/`.

## Current Code Shape
- `main.py` loads the CSV, normalizes features using training-set statistics, builds Python generators, evaluates a naive baseline, then trains several Keras models in sequence.
- Implemented model variants:
  - dense baseline
  - simple GRU
  - GRU with dropout and recurrent dropout
  - stacked GRU
  - bidirectional GRU
- The script is currently written as a single file. Prefer incremental refactoring over large rewrites.

## Environment
- Python project with dependencies pinned in `requirements.txt`.
- Main runtime libraries are `tensorflow`, `numpy`, and `matplotlib`.
- The dependency list appears broader than this repo currently needs. Do not remove packages blindly unless the task explicitly requires dependency cleanup.

## Run Commands
- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Run the training script:
  - `python main.py`

## Working Rules For This Repo
- Preserve the local-data workflow. Do not replace `jena_climate_2009_2016.csv` with remote downloads unless explicitly requested.
- Assume training is expensive. When making code changes, prefer validating with targeted checks before running the full training pipeline.
- Keep reproducibility in mind. This repo already seeds NumPy and TensorFlow in `main()`.
- Preserve existing dataset split semantics unless the task is specifically about experimentation.
- Keep generated artifacts out of the code path. New plots, logs, or checkpoints should go to a dedicated folder instead of the repo root.

## Code Change Guidance
- Prefer small, explicit functions over adding more logic directly into `main()`.
- If adding experiments, isolate hyperparameters as module-level constants or a config structure.
- If saving models or histories, use stable filenames and avoid overwriting unrelated outputs silently.
- Maintain compatibility with the existing TensorFlow/Keras API used in this repo.
- Comments and user-facing text should use UTF-8-safe Russian or English consistently. Some existing comments/output show encoding corruption; avoid introducing more mojibake.

## Validation Guidance
- For nontrivial changes, validate at the smallest reasonable level first:
  - import sanity
  - data loading and preprocessing
  - one batch from the generator
  - model build/summary
- Only run full end-to-end training when the task requires it.

## File Conventions
- Source: `main.py`
- Data: `jena_climate_2009_2016.csv`
- Raw materials: `materials/`
- Output charts: `graphics/`

## Notes For Future Work
- Good next refactors, if requested:
  - split data utilities, model builders, and training loop into separate modules
  - add CLI flags for selecting a model and epoch count
  - save training histories and plots deterministically
  - add a lightweight smoke test for generator shapes and model compilation
