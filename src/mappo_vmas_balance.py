"""
MAPPO + VMAS Balance — pure BenchMARL 1.5.2 defaults.

No hyperparameter overrides. Only infrastructure settings (device, logger, save
path) are set so that training runs on GPU and results are persisted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "BenchMARL"))

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = ExperimentConfig.get_from_yaml()
    config.train_device = "cuda"
    config.loggers = ["csv"]
    config.create_json = True
    config.render = False
    config.save_folder = str(
        Path(__file__).resolve().parents[1] / "results" / "mappo_vmas_balance"
    )
    config.checkpoint_interval = 600_000
    config.checkpoint_at_end = True

    experiment = Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=config,
    )
    experiment.run()
