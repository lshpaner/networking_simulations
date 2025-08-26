# plotting.py
# Matplotlib plotting helpers

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_phase_transition_png(df, out_path="figure_phase_transition.png"):
    plt.figure(figsize=(7, 5))
    for n in sorted(
        df["N"].unique(), key=lambda x: {50: 0, 200: 1, 1000: 2}.get(x, 99)
    ):
        g = df[df["N"] == n].sort_values("c")
        plt.plot(g["c"], g["prob_LCC_ge_50"], marker="o", label=f"N={n}")
    plt.axvline(1.0, linestyle="--")
    plt.ylim(0, 1)
    plt.xlim(0.0, 2.0)
    plt.xlabel("Expected degree, c")
    plt.ylabel("Probability LCC ≥ 50%")
    plt.title("Phase transition: systemic networking vs expected degree")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_heatmap_png(df, out_path="figure_heatmap_lambda_p.png"):
    lam_vals = np.sort(df["lambda"].unique())
    p_vals = np.sort(df["p"].unique())
    mat = np.zeros((p_vals.size, lam_vals.size))
    for i, p in enumerate(p_vals):
        for j, lam in enumerate(lam_vals):
            mat[i, j] = df[(df["p"] == p) & (df["lambda"] == lam)][
                "prob_LCC_ge_50"
            ].values[0]
    plt.figure(figsize=(7, 5))
    plt.imshow(
        mat,
        origin="lower",
        extent=[lam_vals.min(), lam_vals.max(), p_vals.min(), p_vals.max()],
        aspect="auto",
    )
    cbar = plt.colorbar()
    cbar.set_label("Probability LCC ≥ 50%")
    plt.xlabel("Encounter rate λ (per hour)")
    plt.ylabel("Connection probability p")
    plt.title(
        "Connectivity probability across λ and p\n(L = 6 min, T = 60 min, N = 200)"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_effectL_png(df, out_path="figure_effect_L.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(df["L"], df["prob_LCC_ge_50"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Conversation length L (minutes)")
    plt.ylabel("Probability LCC ≥ 50%")
    meta = df.iloc[0]
    plt.title(
        f"Shorter conversations increase systemic networking\n(λ = {meta['lambda']}/h, "
        f"p = {meta['p']}, T = {meta['T']} min, N = {int(meta['N'])})"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
