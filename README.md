# AlphaFold2 From Scratch

PyTorch implementation of AlphaFold2-like components built from scratch for study and experimentation. The model code stays as-is in this pass; the main improvement here is the data layer and the repository scaffolding around it.

## Current Status

- The repository has a working model/training prototype and several unit tests for geometry and losses.
- The data pipeline is now separated from notebook cells into reusable scripts, manifests and YAML configs.
- `data/Proteinas_secuencias.csv` is a checked-in 50-target Foldbench subset exported from Colab.
- Full end-to-end training from scratch has not been validated in this repository yet.

## What Changed In This Cleanup

- `data/download_data.sh`: bash downloader for the Foldbench subset with target-file or CSV input.
- `data/preproces_data.py`: CLI to build or normalize the manifest CSV and emit a YAML summary plus target list.
- `data/dataloaders.py`: dataset can now load from either a manifest CSV or the raw JSON/MSA/mmCIF folders.
- `data/visualize_data.py`: small inspection CLI for manifest summaries, A3M previews and CA distance maps.
- `config/`: repository configs extracted from the notebook plus a full AlphaFold/OpenFold reference config.

## Repository Layout

- `config/data/foldbench_subset.yaml`: dataset paths, outputs and basic stats.
- `config/experiments/af2_poc.yaml`: the current notebook-sized experiment.
- `config/experiments/alphafold2_full_reference.yaml`: reference hyperparameters from AlphaFold/OpenFold.
- `data/`: manifest tooling, dataset code, collate function and download/inspection utilities.
- `model/`: AlphaFold2-like modules.
- `training/`: training loop, EMA, scheduler and checkpoint helpers.
- `tests/`: current unit tests.
- `notebooks/`: exploratory notebooks and the original training notebook.

## Quickstart

1. Create an environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the checked-in Foldbench subset manifest.

```bash
bash data/download_data.sh --targets-csv data/Proteinas_secuencias.csv
```

3. Rebuild the manifest with local paths.

```bash
python3 -m data.preproces_data \
  --config config/data/foldbench_subset.yaml \
  --json-path data/af_subset/jsons/fb_protein.json \
  --msa-root data/af_subset/foldbench_msas \
  --cif-root data/af_subset/reference_structures
```

4. Inspect the manifest or an MSA.

```bash
python3 -m data.visualize_data manifest-summary --manifest-csv data/Proteinas_secuencias.csv
python3 -m data.visualize_data msa-preview --a3m-path data/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m
```

## Data Workflow

The checked-in CSV is useful as a stable subset definition, but its file paths point to Colab. The intended local workflow is:

1. Use `data/download_data.sh` to fetch the subset.
2. Run `python3 -m data.preproces_data` to rewrite the manifest locally.
3. Feed the resulting CSV to `FoldbenchProteinDataset(manifest_csv=...)`.

The dataset class still supports the old JSON + root-directory flow, but the manifest-first workflow is cleaner and easier to reproduce.

## Config Notes

`config/experiments/af2_poc.yaml` mirrors the notebook:

- `max_msa_seqs: 128`
- `batch_size: 2`
- `epochs: 20`
- `lr: 1e-4`
- `num_evoformer_blocks: 2`
- `num_structure_blocks: 4`

`config/experiments/alphafold2_full_reference.yaml` is a reference document, not a promise that the current code already consumes every field. It includes official architecture, loss and training values collected from the AlphaFold/OpenFold configs.

## Tests

Pytest capture is flaky in this workspace, so a safer invocation is:

```bash
python3 -m pytest --capture=no tests
```

If `biopython` is missing, any test or script that parses mmCIF files will fail until dependencies are installed.

## Docker

Build a small CPU-oriented image with:

```bash
docker build -t alphafold-from-scratch .
```

This image is meant for setup and data tooling, not for serious GPU training.

## License

MIT
