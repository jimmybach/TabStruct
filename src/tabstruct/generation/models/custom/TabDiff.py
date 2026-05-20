import pandas as pd

from src.tabstruct.common.runtime.error.ManualStopError import ManualStopError

from ._official_repo import BaseOfficialRepoGenerator


class TabDiff(BaseOfficialRepoGenerator):
    repo_arg_name = "tabdiff_repo_root"
    python_arg_name = "tabdiff_python_bin"
    repo_env_var = "TABDIFF_REPO_ROOT"
    python_env_var = "TABDIFF_PYTHON_BIN"
    repo_dir_candidates = ("TabDiff", "tabdiff")

    def _train_external_model(self):
        self._run_external_command(
            [
                self.python_bin,
                "main.py",
                "--dataname",
                self.external_dataset_name,
                "--mode",
                "train",
                "--gpu",
                str(self._gpu_index()),
                "--exp_name",
                self.external_exp_name,
                "--no_wandb",
            ],
            stage="training",
        )

    def _sample_external_model(self) -> pd.DataFrame:
        self._run_external_command(
            [
                self.python_bin,
                "main.py",
                "--dataname",
                self.external_dataset_name,
                "--mode",
                "test",
                "--gpu",
                str(self._gpu_index()),
                "--exp_name",
                self.external_exp_name,
                "--num_samples_to_generate",
                str(self.num_samples_per_generation_run),
                "--no_wandb",
            ],
            stage="sampling",
        )

        result_root = self.repo_root / "tabdiff" / "result" / self.external_dataset_name / self.external_exp_name
        sample_candidates = sorted(result_root.glob("**/samples.csv"), key=lambda path: path.stat().st_mtime)
        if not sample_candidates:
            raise ManualStopError(f"No TabDiff samples found under {result_root}")

        return pd.read_csv(sample_candidates[-1])

    @classmethod
    def _define_default_params(cls):
        return {
            "architecture": {},
            "optimization": {},
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        return cls._define_default_params()

    @classmethod
    def _define_single_run_params(cls):
        return cls._define_default_params()

    @classmethod
    def _define_test_params(cls):
        return cls._define_default_params()
