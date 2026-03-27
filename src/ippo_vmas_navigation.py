"""IPPO + VMAS Navigation task."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "third_party" / "BenchMARL"))

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":
    experiment = Experiment(
        task=VmasTask.NAVIGATION.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=ExperimentConfig.get_from_yaml(),
    )
    experiment.run()
