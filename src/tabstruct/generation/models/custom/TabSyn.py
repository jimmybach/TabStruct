import pandas as pd

from ._official_repo import BaseOfficialRepoGenerator


class TabSyn(BaseOfficialRepoGenerator):
    repo_arg_name = "tabsyn_repo_root"
    python_arg_name = "tabsyn_python_bin"
    repo_env_var = "TABSYN_REPO_ROOT"
    python_env_var = "TABSYN_PYTHON_BIN"
    repo_dir_candidates = ("tabsyn", "TabSyn")

    def _train_external_model(self):
        gpu_index = self._gpu_index()
        self._run_external_command(
            [
                self.python_bin,
                "main.py",
                "--dataname",
                self.external_dataset_name,
                "--method",
                "vae",
                "--mode",
                "train",
                "--gpu",
                str(gpu_index),
            ],
            stage="VAE training",
        )
        self._run_external_command(
            [
                self.python_bin,
                "main.py",
                "--dataname",
                self.external_dataset_name,
                "--method",
                "tabsyn",
                "--mode",
                "train",
                "--gpu",
                str(gpu_index),
            ],
            stage="diffusion training",
        )

    def _sample_external_model(self) -> pd.DataFrame:
        save_path = self.repo_root / "synthetic" / self.external_dataset_name / f"{self.external_exp_name}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_external_command(
            [
                self.python_bin,
                "main.py",
                "--dataname",
                self.external_dataset_name,
                "--method",
                "tabsyn",
                "--mode",
                "sample",
                "--gpu",
                str(self._gpu_index()),
                "--save_path",
                str(save_path),
            ],
            stage="sampling",
        )

        return pd.read_csv(save_path)

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
