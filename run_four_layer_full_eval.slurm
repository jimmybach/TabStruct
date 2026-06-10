#!/bin/bash
#SBATCH --job-name=four-layer-tabstruct
#SBATCH --output=/sciclone/scr20/jpbach/slurm_logs/%x-%A_%a.out
#SBATCH --error=/sciclone/scr20/jpbach/slurm_logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=hima
#SBATCH --gres=gpu:1
#SBATCH --array=0-404

set -euo pipefail

module load miniforge3/24.9.2-0
module load cuda/12.3
eval "$(conda shell.bash hook)"

WORKROOT="/sciclone/scr20/jpbach"
TABSTRUCT_REPO="$HOME/TabStruct"
TABSTRUCT_PYTHON="/sciclone/scr10/jpbach/conda-envs/tabstruct/bin/python"

FOUR_LAYER_ROOT="${FOUR_LAYER_ROOT:-$HOME/four_layer_datasets_csv_only/four_layer_datasets_csv_only}"
FOUR_LAYER_ROOT_CANDIDATES=(
  "$FOUR_LAYER_ROOT"
  "/sciclone/scr20/jpbach/four_layer_datasets_csv_only/four_layer_datasets_csv_only"
  "/sciclone/home/jpbach/four_layer_datasets_csv_only/four_layer_datasets_csv_only"
)
FOUR_LAYER_DATA_ROOT="$FOUR_LAYER_ROOT/data"
FOUR_LAYER_INFO_ROOT="$FOUR_LAYER_ROOT/Info"
FOUR_LAYER_STAGE_ROOT="$FOUR_LAYER_ROOT/staged"

export TMPDIR="$WORKROOT/tmp"
export PIP_CACHE_DIR="$WORKROOT/pip-cache"
export UV_CACHE_DIR="$WORKROOT/uv-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$WORKROOT/slurm_logs" "$FOUR_LAYER_DATA_ROOT" "$FOUR_LAYER_INFO_ROOT" "$FOUR_LAYER_STAGE_ROOT"

# The four-layer benchmark names datasets by the filename stem, e.g.
# fl4_TrackA_Controlled_SCM_noshift_er_s11
for candidate_root in "${FOUR_LAYER_ROOT_CANDIDATES[@]}"; do
  if [[ -d "$candidate_root" ]]; then
    FOUR_LAYER_ROOT="$candidate_root"
    break
  fi
done

FOUR_LAYER_DATA_ROOT="$FOUR_LAYER_ROOT/data"
FOUR_LAYER_INFO_ROOT="$FOUR_LAYER_ROOT/Info"

mapfile -t SOURCE_CSVS < <(
  find "$FOUR_LAYER_ROOT" -type f -name '*.csv' \
    ! -path "$FOUR_LAYER_INFO_ROOT/*" \
    ! -path "$FOUR_LAYER_STAGE_ROOT/*" \
    ! -path "$FOUR_LAYER_DATA_ROOT/*" \
    | sort
)

NUM_DATASETS=${#SOURCE_CSVS[@]}
if (( NUM_DATASETS == 0 )); then
  echo "No four-layer CSV datasets found under $FOUR_LAYER_ROOT" >&2
  exit 1
fi

echo "Discovered $NUM_DATASETS four-layer datasets under $FOUR_LAYER_ROOT"
printf '  %s\n' "${SOURCE_CSVS[@]:0:5}"

TABSYN_TASKS=$NUM_DATASETS
TABDIFF_TASKS=$NUM_DATASETS

TASK_ID=$SLURM_ARRAY_TASK_ID
if (( TASK_ID < TABSYN_TASKS )); then
  MODEL="tabsyn"
  DATASET_INDEX=$TASK_ID
elif (( TASK_ID < TABSYN_TASKS + TABDIFF_TASKS )); then
  MODEL="tabdiff"
  DATASET_INDEX=$((TASK_ID - TABSYN_TASKS))
else
  MODEL="tabddpm"
  DATASET_INDEX=$((TASK_ID - TABSYN_TASKS - TABDIFF_TASKS))
fi

if (( DATASET_INDEX < 0 || DATASET_INDEX >= NUM_DATASETS )); then
  echo "Task $TASK_ID is outside the dataset range ($NUM_DATASETS datasets)." >&2
  exit 0
fi

SOURCE_CSV="${SOURCE_CSVS[$DATASET_INDEX]}"
rel_source="${SOURCE_CSV#"$FOUR_LAYER_ROOT"/}"
rel_source="${rel_source%.csv}"
rel_source="${rel_source//-/_}"
rel_source="${rel_source//\//_}"
DATASET_KEY="fl4_${rel_source}"
STAGE_DATASET_DIR="$FOUR_LAYER_STAGE_ROOT/$DATASET_KEY"
STAGE_CSV="$STAGE_DATASET_DIR/$DATASET_KEY.csv"
METADATA_JSON="$FOUR_LAYER_INFO_ROOT/$DATASET_KEY.json"

TABSYN_OUTPUT_ROOT="$WORKROOT/tabsyn_outputs"
TABSYN_SYN_CSV="$TABSYN_OUTPUT_ROOT/$DATASET_KEY/tabsyn.csv"

TABDIFF_ROOT="$HOME/TabDiff/tabdiff/result"
TABDIFF_SYN_CSV="$TABDIFF_ROOT/$DATASET_KEY/learnable_schedule/2000/samples.csv"

cd "$TABSTRUCT_REPO"

echo "Running model=$MODEL dataset=$DATASET_KEY"
echo "Source csv: $SOURCE_CSV"
echo "Stage dir: $STAGE_DATASET_DIR"
echo "Host: $(hostname)"
nvidia-smi || true

"$TABSTRUCT_PYTHON" -c "import torch, wandb, lightning; print('tabstruct env ok')"
"$TABSTRUCT_PYTHON" -c "import torch; print('cuda available:', torch.cuda.is_available()); print('devices:', torch.cuda.device_count())"

unset PYTHONPATH

prepare_stage_dataset() {
  local source_csv="$1"
  local dataset_key="$2"
  local stage_dataset_dir="$3"
  local metadata_json="$4"
  local stage_csv="$5"
  local data_root="$6"
  local info_root="$7"

  SOURCE_CSV="$source_csv" \
  DATASET_KEY="$dataset_key" \
  STAGE_DATASET_DIR="$stage_dataset_dir" \
  METADATA_JSON="$metadata_json" \
  STAGE_CSV="$stage_csv" \
  DATA_ROOT="$data_root" \
  INFO_ROOT="$info_root" \
  "$TABSTRUCT_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

source_csv = Path(os.environ["SOURCE_CSV"])
dataset_key = os.environ["DATASET_KEY"]
stage_dataset_dir = Path(os.environ["STAGE_DATASET_DIR"])
metadata_json = Path(os.environ["METADATA_JSON"])
stage_csv = Path(os.environ["STAGE_CSV"])
data_root = Path(os.environ["DATA_ROOT"])
info_root = Path(os.environ["INFO_ROOT"])

stage_dataset_dir.mkdir(parents=True, exist_ok=True)
info_root.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source_csv)
columns = df.columns.tolist()

if "target" in columns:
    target_col = "target"
elif "E" in columns:
    target_col = "E"
else:
    target_col = columns[-1]

feature_cols = [c for c in columns if c != target_col]
target_series = df[target_col]
num_classes = int(target_series.nunique(dropna=False))

if pd.api.types.is_float_dtype(target_series) and num_classes > 20:
    task = "regression"
    task_type = "regression"
else:
    task = "classification"
    task_type = "multiclass" if num_classes > 2 else "binclass"

num_col_idx = [i for i, col in enumerate(columns) if col in feature_cols and pd.api.types.is_numeric_dtype(df[col])]
cat_col_idx = [i for i, col in enumerate(columns) if col in feature_cols and not pd.api.types.is_numeric_dtype(df[col])]
target_col_idx = [columns.index(target_col)]

if stage_csv.exists() or stage_csv.is_symlink():
    stage_csv.unlink()
try:
    stage_csv.symlink_to(source_csv.resolve())
except OSError:
    stage_csv.write_bytes(source_csv.read_bytes())

metadata = {
    "name": dataset_key,
    "task": task,
    "task_type": task_type,
    "target_col": target_col,
    "target_col_idx": target_col_idx,
    "header": "infer",
    "column_names": columns,
    "num_col_idx": num_col_idx,
    "cat_col_idx": cat_col_idx,
    "file_type": "csv",
    "data_path": str(Path("data") / dataset_key / f"{dataset_key}.csv"),
    "test_path": None,
    "num_classes": num_classes,
}

with open(metadata_json, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"TARGET_COL={target_col!r}")
print(f"TASK={task!r}")
print(f"NUM_CLASSES={num_classes!r}")
PY
}

normalize_csv_to_target_schema_copy() {
  local source_csv="$1"
  local model_name="$2"
  local dataset_key="$3"
  local dataset_dir="$4"
  local target_col="$5"

  local normalized_dir="$WORKROOT/normalized_eval/$model_name"
  local normalized_csv="$normalized_dir/$dataset_key.csv"
  mkdir -p "$normalized_dir"

  SOURCE_CSV="$source_csv" \
  DATASET_DIR="$dataset_dir" \
  NORMALIZED_CSV="$normalized_csv" \
  TARGET_COL="$target_col" \
  "$TABSTRUCT_PYTHON" - <<'PY'
from pathlib import Path
import os
import pandas as pd
import sys

source_csv = Path(os.environ["SOURCE_CSV"])
dataset_dir = Path(os.environ["DATASET_DIR"])
normalized_csv = Path(os.environ["NORMALIZED_CSV"])
target_col = os.environ["TARGET_COL"]

schema_csv = dataset_dir / f"{dataset_dir.name}.csv"
if not schema_csv.exists():
    csv_candidates = [p for p in sorted(dataset_dir.glob("*.csv")) if p.name not in {"train.csv", "test.csv"}]
    if len(csv_candidates) != 1:
        raise ValueError(f"Expected exactly one dataset CSV in {dataset_dir}, found {len(csv_candidates)}")
    schema_csv = csv_candidates[0]

real_df = pd.read_csv(schema_csv)
synthetic_df = pd.read_csv(source_csv)

expected_cols = list(real_df.columns)
feature_cols = [c for c in expected_cols if c != target_col]
normalized_cols = feature_cols + ["target"]

synthetic_df = synthetic_df.loc[:, ~synthetic_df.columns.astype(str).str.startswith("Unnamed:")]

if target_col in synthetic_df.columns and "target" in synthetic_df.columns:
    synthetic_df = synthetic_df.drop(columns=[target_col])
elif target_col in synthetic_df.columns and "target" not in synthetic_df.columns:
    synthetic_df = synthetic_df.rename(columns={target_col: "target"})

allowed_cols = set(feature_cols) | {"target"}
extra_cols = [c for c in synthetic_df.columns if c not in allowed_cols]
if extra_cols:
    synthetic_df = synthetic_df.drop(columns=extra_cols)

missing_cols = [c for c in feature_cols if c not in synthetic_df.columns]
if missing_cols:
    raise ValueError(
        f"Synthetic file {source_csv} is missing required columns after normalization: {missing_cols}"
    )

if "target" not in synthetic_df.columns:
    raise ValueError(f"Synthetic file {source_csv} is missing required target column after normalization.")

synthetic_df = synthetic_df[normalized_cols]
synthetic_df.to_csv(normalized_csv, index=False)

print("[DEBUG] schema_csv:", schema_csv, file=sys.stderr)
print("[DEBUG] real schema columns:", expected_cols, file=sys.stderr)
print("[DEBUG] real schema shape:", real_df.shape, file=sys.stderr)
print("[DEBUG] source synthetic path:", source_csv, file=sys.stderr)
print("[DEBUG] source synthetic shape:", pd.read_csv(source_csv).shape, file=sys.stderr)
print("[DEBUG] normalized synthetic path:", normalized_csv, file=sys.stderr)
print("[DEBUG] normalized synthetic columns:", list(synthetic_df.columns), file=sys.stderr)
print("[DEBUG] normalized synthetic shape:", synthetic_df.shape, file=sys.stderr)
PY

  echo "$normalized_csv"
}

eval "$(prepare_stage_dataset "$SOURCE_CSV" "$DATASET_KEY" "$STAGE_DATASET_DIR" "$METADATA_JSON" "$STAGE_CSV" "$FOUR_LAYER_DATA_ROOT" "$FOUR_LAYER_INFO_ROOT")"

if [[ "$MODEL" == "tabddpm" ]]; then
  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabddpm \
    --save_model \
    --dataset "$STAGE_DATASET_DIR" \
    --dataset_root "$FOUR_LAYER_ROOT" \
    --task "$TASK" \
    --test_id 0 \
    --valid_id 0 \
    --valid_size 0.1 \
    --enable_eval_structure \
    --tags hpc four-layer tabddpm full-eval "$DATASET_KEY"

elif [[ "$MODEL" == "tabsyn" ]]; then
  if [[ ! -f "$TABSYN_SYN_CSV" ]]; then
    echo "Missing TabSyn synthetic CSV: $TABSYN_SYN_CSV" >&2
    exit 1
  fi

  NORMALIZED_TABSYN_CSV="$(normalize_csv_to_target_schema_copy "$TABSYN_SYN_CSV" "tabsyn" "$DATASET_KEY" "$STAGE_DATASET_DIR" "$TARGET_COL")"

  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabsyn \
    --generator tabsyn \
    --generator_tags external \
    --eval_only \
    --dataset "$STAGE_DATASET_DIR" \
    --dataset_root "$FOUR_LAYER_ROOT" \
    --task "$TASK" \
    --test_id 0 \
    --valid_id 0 \
    --valid_size 0.1 \
    --synthetic_data_path "$NORMALIZED_TABSYN_CSV" \
    --enable_eval_structure \
    --tags hpc four-layer tabsyn eval full-eval "$DATASET_KEY"

else
  if [[ ! -f "$TABDIFF_SYN_CSV" ]]; then
    echo "Missing TabDiff synthetic CSV: $TABDIFF_SYN_CSV" >&2
    exit 1
  fi

  NORMALIZED_TABDIFF_CSV="$(normalize_csv_to_target_schema_copy "$TABDIFF_SYN_CSV" "tabdiff" "$DATASET_KEY" "$STAGE_DATASET_DIR" "$TARGET_COL")"

  "$TABSTRUCT_PYTHON" -m src.tabstruct.experiment.run_experiment \
    --pipeline generation \
    --model tabdiff \
    --generator tabdiff \
    --generator_tags external \
    --eval_only \
    --dataset "$STAGE_DATASET_DIR" \
    --dataset_root "$FOUR_LAYER_ROOT" \
    --task "$TASK" \
    --test_id 0 \
    --valid_id 0 \
    --valid_size 0.1 \
    --synthetic_data_path "$NORMALIZED_TABDIFF_CSV" \
    --enable_eval_structure \
    --tags hpc four-layer tabdiff eval full-eval "$DATASET_KEY"
fi
