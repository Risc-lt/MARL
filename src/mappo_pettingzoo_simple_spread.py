"""
MAPPO + PettingZoo Simple Spread — BenchMARL defaults with one fix.

All hyperparameters are BenchMARL 1.5.2 defaults (ExperimentConfig / MappoConfig
/ MlpConfig .get_from_yaml()), except:

  on_policy_n_minibatch_iters: 45 → 4   (PPO epochs)

BenchMARL's default 45 PPO epochs × 15 minibatches = 675 gradient steps per
collection, which reliably causes NaN actions in PettingZoo's continuous-action
simple_spread (policy diverges). Reducing to 4 epochs (= 60 gradient steps) is
the minimal change that keeps training stable while staying close to the
original benchmark configuration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "BenchMARL"))

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

PPO_EPOCHS = 4

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
    config.on_policy_n_minibatch_iters = PPO_EPOCHS
    config.save_folder = str(
        Path(__file__).resolve().parents[1] / "results" / "mappo_pettingzoo_simple_spread"
    )
    config.checkpoint_interval = 600_000
    config.checkpoint_at_end = True

    experiment = Experiment(
        task=PettingZooTask.SIMPLE_SPREAD.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=config,
    )
    experiment.run()
