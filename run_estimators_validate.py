import sys
from pathlib import Path
from mutinfo.estimators.knn import KSG, WKL
from sklearn.cross_decomposition import CCA
import pickle
import tqdm
import numpy as np
import pandas as pd
import argparse

import os
import warnings
warnings.filterwarnings("ignore")

def estimate_cca_normal(X, Y, n_components=3):
    cca = CCA(
        n_components=min(
            n_components,
            X.shape[1],
            Y.shape[1],
            X.shape[0] - 1
        )
    )

    X_c, Y_c = cca.fit_transform(X, Y)

    rho = np.array([
        np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        for i in range(cca.n_components)
    ])

    rho_squared = np.clip(rho ** 2, 0, 1 - 1e-10)

    return -0.5 * np.log1p(-rho_squared).sum()

def run_on_benchmark(benchmark_path, out_dir):
    with open(benchmark_path, "rb") as f:
        benchmark = pickle.load(f)
    
    out_name = Path(benchmark_path).stem + "_results.csv"
    estimators = {
        "CCA": estimate_cca_normal,
        "KSG": KSG(k_neighbors=5),
        "WKL": WKL(k_neighbors=5),
    }
    results = []
    for sample_size, samples in benchmark.items():
        for elem in tqdm.tqdm(samples, desc=f"N={sample_size}"):
            X, Y = elem["X"], elem["Y"]
            true_mi = elem["MI"]

            for name, estimator in estimators.items():
                hat_mi = float(estimator(X, Y))

                results.append({
                    "estimator": name,
                    "sample_size": sample_size,
                    "true_mi": true_mi,
                    "hat_mi": hat_mi,
                    "error": hat_mi - true_mi,
                })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, out_name), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сгенерировать и сохранить бенчмарк")
    ap.add_argument("--benchmarks", required=True, help="папка с бенчмарками")
    ap.add_argument("--out", required=True, help="директория для результатов")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for benchmark_name in os.listdir(args.benchmarks):
        benchmark_path = os.path.join(args.benchmarks, benchmark_name)
        run_on_benchmark(benchmark_path, args.out)