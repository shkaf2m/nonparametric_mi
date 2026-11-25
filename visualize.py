import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

def draw_graphs(results_path):
    df = pd.read_csv(results_path)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    mean_hat_mi = df.groupby(["sample_size", "estimator"])["hat_mi"].mean().unstack()
    var = df.groupby(["sample_size", "estimator"])["hat_mi"].var().unstack()

    sample_grid = np.sort(df["sample_size"].unique())
    estimators = np.sort(df["estimator"].unique())
    true_mi = df.groupby("sample_size")["true_mi"].first()
    axes[0].plot(sample_grid, true_mi, label="true MI", color="black", linestyle="--")

    for name in estimators:
        axes[0].plot(mean_hat_mi[name].index.to_numpy(), mean_hat_mi[name].values, label=name, linewidth=5)
        axes[1].plot(mean_hat_mi[name].index.to_numpy(), mean_hat_mi[name].values, label=name, linewidth=5)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    plt.suptitle(f"Бенчмарк: {Path(results_path).name.split("_results.csv")[0]}")
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сгенерировать и сохранить бенчмарк")
    ap.add_argument("--results", required=True, help="папка с результатами")
    args = ap.parse_args()
    for results_name in os.listdir(args.results):
        results_path = os.path.join(args.results, results_name)
        draw_graphs(results_path)