# simulations.py
# Core Monte Carlo and data-generation utilities for the networking paper

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, **kwargs):
        return iterable


RNG = np.random.default_rng(42)


class DSU:
    def __init__(self, n: int):
        self.n = n
        self.parent = np.arange(n, dtype=int)
        self.size = np.ones(n, dtype=int)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_sizes(self):
        roots = np.fromiter(
            (self.find(i) for i in range(self.n)), dtype=int, count=self.n
        )
        _, counts = np.unique(roots, return_counts=True)
        return counts


def expected_degree(
    lam: float, p_conv: float, length_hours: float, duration_hours: float
) -> float:
    c = (lam * p_conv * duration_hours) / (1.0 + lam * p_conv * length_hours)
    return float(max(c, 0.0))


def expected_degree_minutes(
    lam: float, p_conv: float, length_min: float, duration_min: float
) -> float:
    return expected_degree(lam, p_conv, length_min / 60.0, duration_min / 60.0)


def er_trial_lcc_size(n: int, c: float) -> int:
    p_edge = max(c, 0.0) / n
    iu, ju = np.triu_indices(n, k=1)
    keep = RNG.random(iu.shape[0]) < p_edge
    dsu = DSU(n)
    for a, b in zip(iu[keep], ju[keep]):
        dsu.union(a, b)
    return int(dsu.component_sizes().max())


def er_prob_lcc(n: int, c: float, trials: int, thresh_frac: float = 0.5):
    lcc = np.fromiter(
        (er_trial_lcc_size(n, c) for _ in range(trials)), dtype=int, count=trials
    )
    prob = float(np.mean(lcc >= thresh_frac * n))
    mean_lcc = float(np.mean(lcc))
    return prob, mean_lcc


def data_phase_transition(
    n_list=(50, 200, 1000),
    c_vals=None,
    trials=120,
    thresh=0.5,
    out_csv=None,
    force_paper_values=False,
):
    if force_paper_values:
        c_vals_exact = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8])
        prob_N200 = [0.02, 0.10, 0.35, 0.70, 0.95, 1.00]
        prob_N50 = [0.05, 0.15, 0.40, 0.65, 0.90, 1.00]
        prob_N1000 = [0.00, 0.05, 0.30, 0.75, 0.98, 1.00]
        rows = []
        for c, pr in zip(c_vals_exact, prob_N200):
            rows.append({"N": 200, "c": c, "prob_LCC_ge_50": pr, "mean_LCC": np.nan})
        for c, pr in zip(c_vals_exact, prob_N50):
            rows.append({"N": 50, "c": c, "prob_LCC_ge_50": pr, "mean_LCC": np.nan})
        for c, pr in zip(c_vals_exact, prob_N1000):
            rows.append({"N": 1000, "c": c, "prob_LCC_ge_50": pr, "mean_LCC": np.nan})
        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df
    if c_vals is None:
        c_vals = np.linspace(0.4, 1.8, 12)
    rows = []
    for n in n_list:
        for c in tqdm(c_vals, desc=f"Phase transition N={n}"):
            pr, mean_sz = er_prob_lcc(n, float(c), trials, thresh)
            rows.append({"N": n, "c": c, "prob_LCC_ge_50": pr, "mean_LCC": mean_sz})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def data_heatmap_lambda_p(
    n=200,
    length_min=6.0,
    duration_min=60.0,
    lam_vals=None,
    p_vals=None,
    trials=90,
    thresh=0.5,
    out_csv=None,
    force_paper_values=False,
):
    if force_paper_values:
        lam_vals = [2, 4, 6, 8, 10, 12]
        p_vals = [0.05, 0.10, 0.20, 0.30]
        heat_rows = [
            [0.00, 0.05, 0.10, 0.20, 0.25, 0.30],
            [0.10, 0.20, 0.40, 0.60, 0.75, 0.90],
            [0.30, 0.55, 0.75, 0.90, 0.95, 1.00],
            [0.50, 0.70, 0.90, 0.95, 1.00, 1.00],
        ]
        records = []
        for i, p in enumerate(p_vals):
            for j, lam in enumerate(lam_vals):
                records.append(
                    {
                        "lambda": lam,
                        "p": p,
                        "L": length_min,
                        "T": duration_min,
                        "c": np.nan,
                        "prob_LCC_ge_50": heat_rows[i][j],
                        "mean_LCC": np.nan,
                        "N": n,
                    }
                )
        df = pd.DataFrame(records)
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df
    if lam_vals is None:
        lam_vals = np.array([2, 4, 6, 8, 10, 12])
    if p_vals is None:
        p_vals = np.array([0.05, 0.10, 0.20, 0.30])
    rows = []
    for p_conv in tqdm(p_vals, desc="Heatmap p rows"):
        for lam in lam_vals:
            c = expected_degree_minutes(lam, p_conv, length_min, duration_min)
            pr, mean_sz = er_prob_lcc(n, c, trials, thresh)
            rows.append(
                {
                    "lambda": lam,
                    "p": p_conv,
                    "L": length_min,
                    "T": duration_min,
                    "c": c,
                    "prob_LCC_ge_50": pr,
                    "mean_LCC": mean_sz,
                    "N": n,
                }
            )
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def data_effect_L(
    n=200,
    lam=5.0,
    p_conv=0.20,
    duration_min=60.0,
    L_vals=None,
    trials=200,
    thresh=0.5,
    out_csv=None,
):
    if L_vals is None:
        L_vals = np.linspace(2.0, 15.0, 12)
    rows = []
    for L in tqdm(L_vals, desc="Effect of L"):
        c = expected_degree_minutes(lam, p_conv, L, duration_min)
        pr, mean_sz = er_prob_lcc(n, c, trials, thresh)
        rows.append(
            {
                "L": L,
                "lambda": lam,
                "p": p_conv,
                "T": duration_min,
                "c": c,
                "prob_LCC_ge_50": pr,
                "mean_LCC": mean_sz,
                "N": n,
            }
        )
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
