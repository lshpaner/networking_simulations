# run_examples.py
# Run data generation and plotting

import os
from simulations import data_phase_transition, data_heatmap_lambda_p, data_effect_L
from plotting import plot_phase_transition_png, plot_heatmap_png, plot_effectL_png

os.makedirs("data", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

phase_df = data_phase_transition(
    out_csv="data/data_phase_transition.csv", force_paper_values=True
)
heat_df = data_heatmap_lambda_p(
    out_csv="data/data_heatmap_lambda_p.csv", force_paper_values=True
)
effectL_df = data_effect_L(
    out_csv="data/data_effect_L.csv",
    lam=5.0,
    p_conv=0.20,
    duration_min=60.0,
    trials=400,
)

plot_phase_transition_png(
    phase_df, out_path="paper/figures/figure_phase_transition.png"
)
plot_heatmap_png(heat_df, out_path="paper/figures/figure_heatmap_lambda_p.png")
plot_effectL_png(effectL_df, out_path="paper/figures/figure_effect_L.png")

print("Outputs written to data/ and paper/figures/")
