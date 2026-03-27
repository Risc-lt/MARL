# MARL — Multi-agent Reinforcement Learning

## Project structure

```
MARL/
├── README.md
├── src/                              # Experiment scripts
│   ├── mappo_vmas_balance.py         # MAPPO + VMAS Balance
│   ├── ippo_vmas_navigation.py       # IPPO + VMAS Navigation
│   ├── mappo_pettingzoo_simple_spread.py  # MAPPO + PettingZoo Simple Spread
│   └── ippo_pettingzoo_simple_tag.py      # IPPO + PettingZoo Simple Tag
└── third_party/
    └── BenchMARL/                    # BenchMARL framework (v1.5.2)
```

## Prerequisites

- macOS (Apple Silicon tested) or Linux
- Python 3.11+
- Homebrew (macOS, for system dependencies)

---

## Setup

#### Step 1: Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

#### Step 2: Create a virtual environment

```bash
cd MARL/third_party/BenchMARL
uv venv .venv --python 3.11
source .venv/bin/activate
```

#### Step 3: Install BenchMARL with VMAS and PettingZoo

> **Note:** We install `pettingzoo` (without `[all]`) to avoid heavy C/C++ build
> dependencies (box2d, ALE/Atari, cmake). The MPE tasks (simple_spread, simple_tag,
> etc.) work fine without `[all]`.

```bash
uv pip install -e ".[vmas]" pettingzoo pygame wandb
```

- `pygame` is required by PettingZoo MPE environments (simple_spread, simple_tag, etc.)
- `wandb` is required by the default logger config (alternatively, override with `experiment.loggers='[csv]'`)

This installs:
- **BenchMARL** (v1.5.2) — the benchmarking framework
- **TorchRL** (v0.11.1) — RL library from PyTorch
- **PyTorch** (v2.11.0) — deep learning backend
- **VMAS** (v1.5.2) — Vectorized Multi-Agent Simulator
- **PettingZoo** (v1.25.0) — Multi-agent environment suite
- **Hydra** — configuration management
- **pygame** (v2.6.1) — rendering backend for PettingZoo MPE
- **wandb** (v0.25.1) — Weights & Biases logger

> **Note on swig:** If you later need `pettingzoo[all]` (for Box2D envs like multiwalker),
> install swig first: `brew install swig` (macOS) or `apt install swig` (Linux).

#### Step 4: Verify installation

```bash
python -c "import benchmarl, vmas, pettingzoo; print('All good!')"
```

---

#### Step 5: Run experiments

##### Option A: Run the scripts in `src/`

```bash
# From the MARL root directory, with the venv activated:
python src/mappo_vmas_balance.py
python src/ippo_vmas_navigation.py
python src/mappo_pettingzoo_simple_spread.py
python src/ippo_pettingzoo_simple_tag.py
```

##### Option B: Command line via Hydra CLI

```bash
# From third_party/BenchMARL/:
python benchmarl/run.py algorithm=mappo task=vmas/balance
python benchmarl/run.py algorithm=ippo task=vmas/navigation
python benchmarl/run.py algorithm=mappo task=pettingzoo/simple_spread
python benchmarl/run.py algorithm=ippo task=pettingzoo/simple_tag
```

##### Option C: Run all 4 combos as a benchmark

```python
from benchmarl.benchmark import Benchmark
from benchmarl.algorithms import MappoConfig, IppoConfig
from benchmarl.environments import VmasTask, PettingZooTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig

Benchmark(
    algorithm_configs=[MappoConfig.get_from_yaml(), IppoConfig.get_from_yaml()],
    tasks=[VmasTask.BALANCE.get_from_yaml(), PettingZooTask.SIMPLE_SPREAD.get_from_yaml()],
    seeds={0, 1},
    experiment_config=ExperimentConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
).run_sequential()
```

---

##### Verified test results (12K frames, quick smoke test)

| # | Algorithm | Environment | Task | Mean Return | Status |
|---|-----------|-------------|------|-------------|--------|
| 1 | MAPPO | VMAS | balance | -5.69 | Passed |
| 2 | IPPO | VMAS | navigation | -0.34 | Passed |
| 3 | MAPPO | PettingZoo | simple_spread | -115.02 | Passed |
| 4 | IPPO | PettingZoo | simple_tag | -34.55 | Passed |


---


### Key experiment config options

| Option | Default | Description |
|--------|---------|-------------|
| `experiment.max_n_frames` | `3000000` | Total training frames |
| `experiment.lr` | `0.00005` | Learning rate |
| `experiment.gamma` | `0.99` | Discount factor |
| `experiment.loggers` | `[csv,wandb]` | Logger list: `csv`, `wandb`, `tensorboard`, `mflow` |
| `experiment.render` | `true` | Render during evaluation |
| `experiment.evaluation_interval` | `120000` | Eval frequency (in frames) |
| `experiment.evaluation_episodes` | `10` | Episodes per evaluation |
| `experiment.on_policy_collected_frames_per_batch` | `6000` | Frames per on-policy iteration |
| `experiment.on_policy_n_envs_per_worker` | `10` | Parallel envs for collection |
| `experiment.on_policy_n_minibatch_iters` | `45` | Minibatch training passes |
| `experiment.on_policy_minibatch_size` | `400` | Minibatch size |
| `experiment.sampling_device` | `cpu` | Device for collection (`cpu` or `cuda`) |
| `experiment.train_device` | `cpu` | Device for training (`cpu` or `cuda`) |
| `experiment.save_folder` | `null` | Custom output path (defaults to Hydra output dir) |
| `experiment.checkpoint_interval` | `0` | Checkpoint frequency in frames (0 = disabled) |
| `experiment.checkpoint_at_end` | `false` | Save checkpoint when experiment finishes |
| `seed` | `0` | Random seed |

### Quick smoke-test overrides

```bash
# Append these to any run command for a fast ~12K frame test:
python benchmarl/run.py algorithm=mappo task=vmas/balance \
  experiment.max_n_frames=12000 \
  experiment.evaluation_interval=6000 \
  experiment.loggers='[csv]' \
  experiment.render=false
```

---

## Environment variables

BenchMARL itself does not read any environment variables. The following are consumed
by its underlying libraries and may need to be set depending on your setup:

### Weights & Biases (wandb)

```bash
# Required if using wandb logger (the default). Get your key from https://wandb.ai/authorize
export WANDB_API_KEY="your-api-key"

# Optional: set default entity/project
export WANDB_ENTITY="your-team"
export WANDB_PROJECT="benchmarl"

# Run wandb in offline mode (no internet needed, sync later with `wandb sync`)
export WANDB_MODE="offline"
```

Alternatively, skip wandb entirely by overriding the logger:
```bash
experiment.loggers='[csv]'
```

### PyTorch / CUDA

```bash
# Select which GPU(s) to use
export CUDA_VISIBLE_DEVICES=0

# Use CPU only (also settable via experiment.train_device / experiment.sampling_device)
export CUDA_VISIBLE_DEVICES=""
```

### Hydra

```bash
# Show full Python tracebacks on errors (useful for debugging)
export HYDRA_FULL_ERROR=1
```
