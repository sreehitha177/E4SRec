# Metadata Annotation and Validation

This directory contains the LLM-based metadata pipelines for:

- `composition`
- `vocal`
- `instrument`

Each pipeline supports two workflows:

1. Full-dataset annotation over `shared/top_50k_songs.csv` (typically via SLURM array jobs)
2. Validation against `shared/validation_df.csv` with metrics and plots

## Directory Layout

- `shared/`
  - `top_50k_songs.csv`: full song list for annotation
  - `validation_df.csv`: validation tracks + ground-truth `gene_values`
  - `ordered_song_list.csv`: additional shared list used by related scripts
- `composition/`, `vocal/`, `instrument/`
  - `*_common.py`: feature definitions, prompt builders, Azure OpenAI calls, parallel annotation helpers
  - `annotate_*_features.py`: shard annotation (`annotate`) and shard merge (`merge`)
  - `validate_*_features.py`: validation annotation + evaluation + plots
  - `run_*_features.sh`: sbatch wrappers for full-dataset annotation
  - `output/`: generated CSV outputs
  - `plots/`: validation plots
  - `logs/`: SLURM logs
- `.env`: runtime environment variables

## Environment Setup

Set and export variables from `metadata/.env` before running Python scripts interactively:

```bash
cd /work/pi_dagarwal_umass_edu/project_7/hmagapu
set -a
source metadata/.env
set +a
```

Expected variables:

- `LLM_API_KEY`
- `AZURE_OPENAI_API_BASE`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_MODEL` (deployment name, optionally prefixed with `azure/`)

## Full Annotation (SLURM)

The sbatch scripts now use:

- shared input list: `metadata/shared/top_50k_songs.csv`
- validation seed annotations from each domain's `output/validation_*_annotations.csv`
- all features by default (no `--features` flag required)
- `--max-workers 10`

Submit jobs:

```bash
sbatch metadata/composition/run_composition_features.sh
sbatch metadata/vocal/run_vocal_features.sh
sbatch metadata/instrument/run_instrument_features.sh
```

### Merge Shards

After array jobs finish, merge shards into one CSV per domain:

```bash
python metadata/composition/annotate_composition_features.py merge \
  --song-list-path metadata/shared/top_50k_songs.csv \
  --output-csv metadata/composition/output/composition_annotations.csv \
  --seed-annotations-csv metadata/composition/output/validation_composition_annotations.csv

python metadata/vocal/annotate_vocal_features.py merge \
  --song-list-path metadata/shared/top_50k_songs.csv \
  --output-csv metadata/vocal/output/vocal_annotations.csv \
  --seed-annotations-csv metadata/vocal/output/validation_vocal_annotations.csv

python metadata/instrument/annotate_instrument_features.py merge \
  --song-list-path metadata/shared/top_50k_songs.csv \
  --output-csv metadata/instrument/output/instrument_annotations.csv \
  --seed-annotations-csv metadata/instrument/output/validation_instrument_annotations.csv
```

## Validation Workflow

Validation scripts run annotation on `shared/validation_df.csv`, compare with MGPHot labels, and generate:

- per-feature Spearman rho
- per-feature MAE
- bias table (predicted mean minus ground-truth mean)
- heatmap plot
- distribution plot

Run composition validation:

```bash
cd /work/pi_dagarwal_umass_edu/project_7/hmagapu
set -a
source metadata/.env
set +a
python metadata/composition/validate_composition_features.py \
  --validation-csv metadata/shared/validation_df.csv \
  --output-csv metadata/composition/output/validation_composition_annotations.csv \
  --max-workers 10
```

Run vocal validation:

```bash
cd /work/pi_dagarwal_umass_edu/project_7/hmagapu
set -a
source metadata/.env
set +a
python metadata/vocal/validate_vocal_features.py \
  --validation-csv metadata/shared/validation_df.csv \
  --output-csv metadata/vocal/output/validation_vocal_annotations.csv \
  --max-workers 10
```

Run instrument validation:

```bash
cd /work/pi_dagarwal_umass_edu/project_7/hmagapu
set -a
source metadata/.env
set +a
python metadata/instrument/validate_instrument_features.py \
  --validation-csv metadata/shared/validation_df.csv \
  --output-csv metadata/instrument/output/validation_instrument_annotations.csv \
  --max-workers 10
```

### Plot Regeneration Only

If annotation CSV already exists, skip LLM calls and regenerate metrics/plots:

```bash
python metadata/composition/validate_composition_features.py --plots-only
python metadata/vocal/validate_vocal_features.py --plots-only
python metadata/instrument/validate_instrument_features.py --plots-only
```

## Output Files

Per domain (`composition`, `vocal`, `instrument`):

- Full annotation shards:
  - `output/<domain>_annotations_shard_<rank>.csv`
- Full merged annotations:
  - `output/<domain>_annotations.csv`
- Validation annotations:
  - `output/validation_<domain>_annotations.csv`
- Validation plots:
  - `plots/validation_<domain>_metrics_heatmap.png`
  - `plots/validation_<domain>_distributions.png`

## Notes

- Annotation scripts retry failed requests with exponential backoff, but heavy parallelism can still hit rate limits (`429 too_many_requests`).
- If rate limits are frequent, lower `--max-workers`.
