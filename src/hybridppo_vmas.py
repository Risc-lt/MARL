"""HybridPPO runner for VMAS Balance and Navigation."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_ROOT = PROJECT_ROOT / "third_party" / "BenchMARL"
sys.path.insert(0, str(BENCHMARL_ROOT))

from benchmarl.algorithms import HybridppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


TASK_MAP = {
    "balance": VmasTask.BALANCE,
    "navigation": VmasTask.NAVIGATION,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HybridPPO on a VMAS task with explicit alpha and seed."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_MAP.keys()),
        default="balance",
        help="VMAS task to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid critic mixing weight in [0, 1].",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=3_000_000,
        help="Maximum number of training frames.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional root directory for experiment outputs.",
    )
    return parser.parse_args()


def _default_save_dir(task: str, alpha: float, seed: int) -> Path:
    alpha_tag = f"{alpha:.2f}".replace(".", "p")
    return PROJECT_ROOT / "runs" / f"hybridppo_{task}_alpha{alpha_tag}_seed{seed}"


if __name__ == "__main__":
    args = _parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}.")

    algorithm_config = HybridppoConfig.get_from_yaml()
    algorithm_config.alpha = args.alpha

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = args.max_frames
    experiment_config.loggers = ["csv"]
    experiment_config.render = False

    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir is not None
        else _default_save_dir(args.task, args.alpha, args.seed)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(save_dir)

    experiment = Experiment(
        task=TASK_MAP[args.task].get_from_yaml(),
        algorithm_config=algorithm_config,
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=experiment_config,
    )
    experiment.run()

    print(f"Finished run saved under: {experiment.folder_name}")
