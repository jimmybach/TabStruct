import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from src.tabstruct.common.data.DataHelper import DataHelper
from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError
from src.tabstruct.common.runtime.log.TerminalIO import TerminalIO

from ..BaseGenerator import BaseMixedGenerator


class BaseOfficialRepoGenerator(BaseMixedGenerator):
    repo_arg_name = None
    python_arg_name = None
    repo_env_var = None
    python_env_var = None
    repo_dir_candidates = ()

    def __init__(self, args):
        super().__init__(args)

        self.repo_root = self._resolve_repo_root()
        self.python_bin = self._resolve_python_bin()
        self.external_dataset_name = self._build_external_dataset_name()
        self.external_exp_name = self._build_external_exp_name()

    def _fit(self, data_module):
        self.prepare_data(data_module)
        training_df = self._build_external_training_dataframe(data_module)
        self._prepare_external_dataset(training_df)
        self._train_external_model()

    def _model_generate(self):
        synthetic_df = self._sample_external_model()
        synthetic_df = self._align_generated_dataframe(synthetic_df)
        return self._convert_original_dataframe_to_mixed_arrays(synthetic_df)

    def _resolve_repo_root(self) -> Path:
        repo_root = getattr(self.args, self.repo_arg_name, None) or os.environ.get(self.repo_env_var)
        if repo_root is not None:
            repo_root = Path(repo_root).expanduser().resolve()
            if not repo_root.exists():
                raise ManualStopError(f"{self.args.model} repository does not exist: {repo_root}")
            return repo_root

        search_roots = [
            Path.cwd(),
            Path.cwd().parent,
            Path("/content"),
            Path.home(),
        ]
        for search_root in search_roots:
            for candidate_name in self.repo_dir_candidates:
                candidate = (search_root / candidate_name).resolve()
                if (candidate / "main.py").exists():
                    TerminalIO.print(
                        f"Auto-detected {self.args.model} repository at {candidate}",
                        color=TerminalIO.OKBLUE,
                    )
                    return candidate

        raise ManualStopError(
            f"{self.args.model} requires the official repository path. "
            f"Set --{self.repo_arg_name} or {self.repo_env_var}. "
            f"In Colab, cloning the repo to /content/{self.repo_dir_candidates[0]} is also supported."
        )

    def _resolve_python_bin(self) -> str:
        python_bin = getattr(self.args, self.python_arg_name, None) or os.environ.get(self.python_env_var)
        return python_bin or sys.executable

    def _build_external_dataset_name(self) -> str:
        dataset_name = Path(self.args.dataset).name.replace("-", "_")
        run_id = wandb.run.id if wandb.run is not None else "offline"
        return f"tabstruct_{self.args.model}_{dataset_name}_{run_id}"

    def _build_external_exp_name(self) -> str:
        run_id = wandb.run.id if wandb.run is not None else "offline"
        return f"tabstruct_{self.args.model}_{run_id}"

    def _build_external_training_dataframe(self, data_module) -> pd.DataFrame:
        train_original = DataHelper.recover_original_data(self.args, data_module.X_train, data_module.y_train)
        valid_original = DataHelper.recover_original_data(self.args, data_module.X_valid, data_module.y_valid)

        train_df = pd.concat(
            [
                pd.concat([train_original["X_original"], train_original["y_original"]], axis=1),
                pd.concat([valid_original["X_original"], valid_original["y_original"]], axis=1),
            ],
            axis=0,
            ignore_index=True,
        )

        ordered_cols = self.args.full_feature_col_list_original + [self.args.full_target_col_original]
        return train_df[ordered_cols]

    def _prepare_external_dataset(self, train_df: pd.DataFrame):
        dataset_dir = self.repo_root / "data" / self.external_dataset_name
        info_dir = self.repo_root / "data" / "Info"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)

        data_path = dataset_dir / f"{self.external_dataset_name}.csv"
        train_df.to_csv(data_path, index=False)

        column_names = train_df.columns.tolist()
        num_col_idx = [
            idx
            for idx, col in enumerate(column_names)
            if col in self.args.num_feature_col_list_original
        ]
        cat_col_idx = [
            idx
            for idx, col in enumerate(column_names)
            if col not in self.args.num_feature_col_list_original and col != self.args.full_target_col_original
        ]
        target_col_idx = [column_names.index(self.args.full_target_col_original)]

        info = {
            "name": self.external_dataset_name,
            "task_type": "regression" if self.args.task == "regression" else "binclass",
            "header": "infer",
            "column_names": None,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "file_type": "csv",
            "data_path": str(Path("data") / self.external_dataset_name / f"{self.external_dataset_name}.csv"),
            "test_path": None,
        }

        with open(info_dir / f"{self.external_dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)

        self._run_external_command(
            [self.python_bin, "process_dataset.py", "--dataname", self.external_dataset_name],
            stage="dataset preprocessing",
        )

    def _align_generated_dataframe(self, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        ordered_cols = self.args.full_feature_col_list_original + [self.args.full_target_col_original]
        missing_cols = [col for col in ordered_cols if col not in synthetic_df.columns]
        if missing_cols:
            raise ManualStopError(
                f"{self.args.model} generated samples missing columns {missing_cols}. "
                f"Available columns: {list(synthetic_df.columns)}"
            )

        return synthetic_df[ordered_cols]

    def _convert_original_dataframe_to_mixed_arrays(self, synthetic_df: pd.DataFrame) -> dict:
        X_syn_original = synthetic_df[self.args.full_feature_col_list_original].copy(deep=True)
        y_syn_original = synthetic_df[[self.args.full_target_col_original]].copy(deep=True)

        X_syn = X_syn_original.copy(deep=True)
        y_syn = y_syn_original.copy(deep=True)

        for feature_scaler in self.args.feature_scaler_list:
            X_syn = feature_scaler.transform(X_syn)
        for target_scaler in self.args.target_scaler_list:
            y_syn = target_scaler.transform(y_syn)

        X_syn = X_syn[self.args.full_feature_col_list_processed].to_numpy().astype(np.float32)
        y_syn = y_syn[self.args.full_target_col_processed].to_numpy()

        num_feature_count = len(self.args.num_feature_col_list_processed)
        syn_num = X_syn[:, :num_feature_count]
        syn_cat = X_syn[:, num_feature_count:]

        if self.args.task == "regression":
            syn_num = np.concatenate([syn_num, y_syn.reshape(-1, 1)], axis=1)
        else:
            syn_cat = np.concatenate([syn_cat, y_syn.reshape(-1, 1)], axis=1).astype(np.int64)

        return {
            "syn_num": syn_num,
            "syn_cat": syn_cat,
        }

    def _gpu_index(self) -> int:
        if self.args.device.startswith("cuda"):
            if ":" in self.args.device:
                return int(self.args.device.split(":", maxsplit=1)[1])
            return 0
        return 0

    def _run_external_command(self, cmd: list[str], stage: str):
        TerminalIO.print(
            f"Running {self.args.model} {stage}: {' '.join(str(part) for part in cmd)}",
            color=TerminalIO.OKBLUE,
        )
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            check=False,
        )
        if result.returncode != 0:
            raise ManualStopError(
                f"{self.args.model} failed during {stage} with exit code {result.returncode}."
            )

    def _train_external_model(self):
        raise NotImplementedError

    def _sample_external_model(self) -> pd.DataFrame:
        raise NotImplementedError
