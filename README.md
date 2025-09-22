# KL Divergence Estimation with Neural Networks

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy matplotlib scikit-learn tqdm
```

## Usage

### Basic Usage

Run full experiment experiment with default parameters:

```bash
python main.py --rerun
```

**Parameters:**

- `--rerun`: Force regeneration of results (ignore cached data)
- `--runs_T <n>`: Number of runs per T (training iterations) setting
- `--runs_m <n>`: Number of runs per m (network size) setting
- `--its_eval <n>`: Number of evaluation iterations
- `--five_d`: Use 5D problem (default: 2D)
- `--fixed_m <n>`: Fixed m for T sweep
- `--fixed_T <n>`: Fixed T for m sweep

### Configuration

The experiment configurations can be modified in `experiments.py`:

```python
def make_default_config_2d() -> SweepConfig:
    return SweepConfig(
        vary_T_values=[100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
        vary_m_values=[5, 10, 50, 100, 500, 1_000],
        fixed_m_for_T=50,           # Network size when varying T
        fixed_T_for_m=500_000,      # Training iterations when varying m
        runs_per_setting_T=10,      # Runs per T value
        runs_per_setting_m=10,      # Runs per m value
        its_eval=5_000,             # Evaluation iterations
    )
```
