"""IPPO + VMAS Balance — phase-2 baseline runner.

Mirrors src/mappo_vmas_balance.py. Parameterised by --seed so the three Balance
baseline seeds (5, 56, 567) can run in parallel without colliding on save path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "BenchMARL"))

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        help="train/sampling device (cpu|cuda|mps). Phase 1 used cpu for IPPO.")
    parser.add_argument("--max_frames", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=10,
                        help="on_policy_n_envs_per_worker. Reduce to 4 if training NaNs (per DIVISION_OF_WORK).")
    args = parser.parse_args()

    config = ExperimentConfig.get_from_yaml()
    config.train_device = args.device
    config.sampling_device = args.device
    config.max_n_frames = args.max_frames
    config.on_policy_n_envs_per_worker = args.n_envs
    config.loggers = ["csv"]
    config.create_json = True
    config.render = False
    save_folder = Path(__file__).resolve().parents[1] / "results" / f"ippo_vmas_balance_seed{args.seed}"
    save_folder.mkdir(parents=True, exist_ok=True)
    config.save_folder = str(save_folder)
    config.checkpoint_interval = 600_000
    config.checkpoint_at_end = True

    experiment = Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=config,
    )
    experiment.run()
