# Networking Simulations: Monte Carlo and Random Graph Modeling

This repository contains the code and paper for:

**Monte Carlo Simulation of Networking Dynamics in Social Events: A Random Graph Approach**

The project demonstrates how systemic networking emerges as a phase transition in Erdős–Rényi graphs, using both Monte Carlo simulations and exact illustrative values for reproducibility.

## Contents
- `src/`: Python source code for simulations and plotting
- `paper/`: LaTeX source of the manuscript and generated figures
- `data/`: CSV outputs from simulation runs

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt

2. Run the example figure generation:

```python
python src/run_examples.py
```

3. Outputs:

- CSV files in `data/`
- Figures in `paper/figures/`

## Reproducibility

- Toggle between exact paper values and true **Monte Carlo simulations** with the `force_paper_values` flag in `src/simulations.py`.

## License


---

### `src/simulations.py`
Put the **full script we just finalized** (with `force_paper_values` toggles) in here.

---

### `src/plotting.py`
Move the three plotting functions (`plot_phase_transition_png`, `plot_heatmap_png`, `plot_effectL_png`) here.

---

### `src/run_examples.py`

```python
from simulations import data_phase_transition, data_heatmap_lambda_p, data_effect_L
from plotting import plot_phase_transition_png, plot_heatmap_png, plot_effectL_png

phase_df = data_phase_transition(force_paper_values=True)
heat_df  = data_heatmap_lambda_p(force_paper_values=True)
effectL_df = data_effect_L()

plot_phase_transition_png(phase_df, out_path="../paper/figures/figure_phase_transition.png")
plot_heatmap_png(heat_df, out_path="../paper/figures/figure_heatmap_lambda_p.png")
plot_effectL_png(effectL_df, out_path="../paper/figures/figure_effect_L.png")
```