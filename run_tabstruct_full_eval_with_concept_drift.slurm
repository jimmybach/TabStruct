#!/bin/bash
#SBATCH --job-name=tabstruct-full-bench
#SBATCH --output=/sciclone/scr20/jpbach/slurm_logs/%x-%A_%a.out
#SBATCH --error=/sciclone/scr20/jpbach/slurm_logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=hima
#SBATCH --gres=gpu:1
#SBATCH --array=0-41

set -euo pipefail

module load miniforge3/24.9.2-0
module load cuda/12.3
eval "$(conda shell.bash hook)"

WORKROOT="/sciclone/scr20/jpbach"
TABSTRUCT_REPO="$HOME/TabStruct"
DATA_ROOT="$HOME/TabStruct/data"
CONCEPT_DRIFT_ROOT="$HOME/data_concept_drift"
TABSTRUCT_PYTHON="/sciclone/scr10/jpbach/conda-envs/tabstruct/bin/python"

export TMPDIR="$WORKROOT/tmp"
export PIP_CACHE_DIR="$WORKROOT/pip-cache"
export UV_CACHE_DIR="$WORKROOT/uv-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$WORKROOT/slurm_logs"

MODELS=(tabsyn tabdiff tabddpm)
DATASET_PATHS=(
  "$DATA_ROOT/diabetes"
  "$DATA_ROOT/house"
  "$DATA_ROOT/income"
  "$DATA_ROOT/sick"
  "$DATA_ROOT/us_location"
  "$CONCEPT_DRIFT_ROOT/P(Y)/credit"
  "$CONCEPT_DRIFT_ROOT/P(Y)/bank_marketing"
  "$CONCEPT_DRIFT_ROOT/P(Y)/adult"
  "$CONCEPT_DRIFT_ROOT/P(X)/gas_sensor"
  "$CONCEPT_DRIFT_ROOT/P(X)/agrawal"
  "$CONCEPT_DRIFT_ROOT/P(X)/gaussian_mixture"
  "$CONCEPT_DRIFT_ROOT/P(YX)/stagger"
  "$CONCEPT_DRIFT_ROOT/P(YX)/hyperplane"
  "$CONCEPT_DRIFT_ROOT/P(YX)/sine"
)

NUM_DATASETS=${#DATASET_PATHS[@]}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

MODEL="${MODELS[$MODEL_INDEX]}"
DATASET_PATH="${DATASET_PATHS[$DATASET_INDEX]}"
DATASET_KEY="$(basename "$DATASET_PATH")"

TABSYN_OUTPUT_ROOT="$WORKROOT/tabsyn_outputs"
TABSYN_SYN_CSV="$TABSYN_OUTPUT_ROOT/$DATASET_KEY/tabsyn.csv"

TABDIFF_ROOT="/sciclone/scr20/jpbach/TabDiff/tabdiff/result"
TABDIFF_SYN_CSV="$TABDIFF_ROOT/$DATASET_KEY/learnable_schedule/2000/samples.csv"

cd "$TABSTRUCT_REPO"

echo "Running model=$MODEL dataset_key=$DATASET_KEY"
echo "Dataset path: $DATASET_PATH"
echo "Host: $(hostname)"
nvidia-smi || true

"$TABSTRUCT_PYTHON" -c "import torch, wandb, lightning; print('tabstruct env ok')"
"$TABSTRUCT_PYTHON" -c "import torch; print('cuda available:', torch.cuda.is_available()); print('devices:', torch.cuda.device_count())"

unset PYTHONPATH

normalize_csv_to_target_schema_copy() {
  local source_csv="$1"
  local model_name="$2"
  local dataset_key="$3"
  local dataset_dir="$4"

  if [[ ! -d "$dataset_dir" ]]; then
    echo "Missing dataset directory: $dataset_dir" >&2
    exit 1
  fi

  local normalized_dir="$WORKROOT/normalized_eval/$model_name"
  local normalized_csv="$normalized_dir/$dataset_key.csv"
  mkdir -p "$normalized_dir"

  "$TABSTRUCT_PYTHON" - <<PY
from pathlib import Path
import pandas as pd
import sys

source_csv = Path("$source_csv")
dataset_dir = Path("$dataset_dir")
normalized_csv = Path("$normalized_csv")

train_csv = dataset_dir / "train.csv"
test_csv = dataset_dir / "test.csv"

if train_csv.exists():
    schema_csv = train_csv
else:
    csv_candidates = [p for p in sorted(dataset_dir.glob("*.csv")) if p.name not in {"train.csv", "test.csv"}]
    if len(csv_candidates) != 1:
        raise ValueError(
            f"Expected exactly one schema CSV in {dataset_dir} when train.csv is absent, found {len(csv_candidates)}"
        )
    schema_csv = csv_candidates[0]

real_df = pd.read_csv(schema_csv)
synthetic_df = pd.read_csv(source_csv)

expected_cols = list(real_df.columns)
target_col = expected_cols[-1]
normalized_cols = expected_cols[:-1] + ["target"]

synthetic_df = synthetic_df.loc[:, ~synthetic_df.columns.astype(str).str.startswith("Unnamed:")]

if target_col in synthetic_df.columns and "target" in synthetic_df.columns:
    synthetic_df = synthetic_df.drop(columns=[target_col])
elif target_col in synthetic_df.columns and "target" not in synthetic_df.columns:
    synthetic_df = synthetic_df.rename(columns={target_col: "target"})

allowed_cols = set(expected_cols[:-1]) | {"target"}
extra_cols = [c for c in synthetic_df.columns if c not in allowed_cols]
if extra_cols:
    synthetic_df = synthetic_df.drop(columns=extra_cols)

missing_cols = [c for c in expected_cols[:-1] if c not in synthetic_df.columns]
if missing_cols:
    raise ValueError(
        f"Synthetic file {source_csv} is missing required columns after normalization: {missing_cols}"
    )
if "target" not in synthetic_df.columns:
    raise ValueError(f"Synthetic file {source_csv} is missing required target column after normalization.")

synthetic_df = synthetic_df[normalized_cols]
synthetic_df.to_csv(normalized_csv, index=False)

print("Schema CSV:", schema_csv, file=sys.stderr)
print("Real schema columns:", expected_cols, file=sys.stderr)
print("Real schema shape:", real_df.shape, file=sys.stderr)
print("Source synthetic path:", source_csv, file=sys.stderr)
print("Source synthetic shape:", pd.read_csv(source_csv).shape, file=sys.stderr)
print("Normalized synthetic path:", normalized_csv, file=sys.stderr)
print("Normalized synthetic columns:", list(synthetic_df.columns), file=sys.stderr)
print("Normalized synthetic shape:", synthetic_df.shape, file=sys.stderr)
PY

  echo "$normalized_csv"
}

if [[ "$MODEL" == "tabddpm" ]]; then
  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabddpm \
    --save_model \
    --dataset "$DATASET_PATH" \
    --dataset_root "$DATA_ROOT" \
    --valid_size 0.1 \
    --enable_eval_structure \
    --tags hpc tabddpm full-eval "$DATASET_KEY"

elif [[ "$MODEL" == "tabsyn" ]]; then
  if [[ ! -f "$TABSYN_SYN_CSV" ]]; then
    echo "Missing TabSyn synthetic CSV: $TABSYN_SYN_CSV" >&2
    exit 1
  fi

  NORMALIZED_TABSYN_CSV="$(normalize_csv_to_target_schema_copy "$TABSYN_SYN_CSV" "tabsyn" "$DATASET_KEY" "$DATASET_PATH")"

  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabsyn \
    --generator tabsyn \
    --generator_tags external \
    --eval_only \
    --dataset "$DATASET_PATH" \
    --dataset_root "$DATA_ROOT" \
    --valid_size 0.1 \
    --synthetic_data_path "$NORMALIZED_TABSYN_CSV" \
    --enable_eval_structure \
    --tags hpc tabsyn eval full-eval "$DATASET_KEY"

else
  if [[ ! -f "$TABDIFF_SYN_CSV" ]]; then
    echo "Missing TabDiff synthetic CSV: $TABDIFF_SYN_CSV" >&2
    exit 1
  fi

  NORMALIZED_TABDIFF_CSV="$(normalize_csv_to_target_schema_copy "$TABDIFF_SYN_CSV" "tabdiff" "$DATASET_KEY" "$DATASET_PATH")"

  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabdiff \
    --generator tabdiff \
    --generator_tags external \
    --eval_only \
    --dataset "$DATASET_PATH" \
    --dataset_root "$DATA_ROOT" \
    --valid_size 0.1 \
    --synthetic_data_path "$NORMALIZED_TABDIFF_CSV" \
    --enable_eval_structure \
    --tags hpc tabdiff eval full-eval "$DATASET_KEY"
fi
