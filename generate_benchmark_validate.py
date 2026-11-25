import argparse
import pickle
import os
import platform
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.special import digamma
from pathlib import Path

"""
Создаём бенчмарки для сравнения оценок взаимной информации
"""


SAMPLE_SIZE_GRID = [110, 250, 500, 1000]
# SAMPLE_SIZE_GRID = [110, 250, 500, 1000, 2500, 5000, 10000]
DIM_GRID = [2, 5, 10, 20, 50, 100]

def generate_normal_sample(
    mi = 4,
    sample_size = 110,
    dim_count = 5,
    depends_dim_count = 3,
    seed = 42
):
    # создаем benchmark с размером выборок sample_size
    rng = np.random.default_rng(seed=seed)
    # X, Y ~ N((I, Sigma, Sigma^T, I))
    if depends_dim_count > dim_count:
        raise ValueError("Error: depends_dim_count > dim_count")
    if depends_dim_count > 0 and mi == 0:
        raise ValueError("Error: mi == 0, but depends_dim_count > 0")
    if mi != 0 and depends_dim_count == 0:
        raise ValueError("Error: mi != 0, but depends_dim_count == 0")

    sample_benchmark = []

    if mi != 0:
        rho = np.sqrt(-np.expm1(-2 * mi / depends_dim_count))
    else:
        rho = 0
    
    # делаем cov(X, Y) = diag(pho, pho, 0, 0, 0, ...)

    sigma_xy = np.zeros((dim_count, dim_count))
    if depends_dim_count > 0:
        sigma_xy[np.arange(depends_dim_count), np.arange(depends_dim_count)] = rho
    
    cov = np.block([
        [np.eye(dim_count), sigma_xy],
        [sigma_xy.T, np.eye(dim_count)]
    ])
    samples = rng.multivariate_normal(
        mean=np.zeros(dim_count + dim_count),
        cov=cov,
        size=sample_size
    )
    X = samples[:, :dim_count]
    Y = samples[:, dim_count:]
    
    return {
        "X": X,
        "Y": Y,
        "MI": mi,
        "seed": seed
    }

def generate_uniform_sample(
    mi = 4,
    sample_size = 110,
    dim_count = 5,
    depends_dim_count = 3,
    seed = 42
):
    # создаем benchmark с размером выборок sample_size
    rng = np.random.default_rng(seed=seed)
    # xi, eta ~ N((I, Sigma, Sigma^T, I))
    # X = F(xi), Y = F(eta) (по компонентно)

    gaussian_dict = generate_normal_sample(mi, sample_size, dim_count, depends_dim_count, seed)
    gaussian_X, gaussian_Y = gaussian_dict["X"], gaussian_dict["Y"]
    # переходим к uniform
    uniform_X = sps.norm(loc=0, scale=1).cdf(gaussian_X)
    uniform_Y = sps.norm(loc=0, scale=1).cdf(gaussian_Y)
    
    return {
        "X": uniform_X,
        "Y": uniform_Y,
        "MI": mi,
        "seed": seed
    }

def generate_loggammaexpon_sample(
    theta = 1,
    sample_size = 110,
    seed = 42
):
    # создаем benchmark с размером выборок sample_size
    rng = np.random.default_rng(seed=seed)

    if theta <= 0:
        raise ValueError("Error: theta must be greater than zero")
    sample_benchmark = []
    exp_X = rng.gamma(
        shape=theta,
        scale=1.0,
        size=sample_size
    )
    
    exp_Y = rng.exponential(
        scale=1.0 / exp_X,
        size=sample_size
    )

    X = np.log(exp_X).reshape(-1, 1)
    Y = np.log(exp_Y).reshape(-1, 1)
    
    MI = digamma(theta + 1) - np.log(theta)

    return {
        "X": X,
        "Y": Y,
        "MI": MI,
        "seed": seed
    }

# def save_benchmark(out_dir, seed=42):
# def generate_normal_benchmark(
#     out_dir,
#     dim_count = 5,
    
#     seed=42
# ):
#     mi_grid = np.linspace(-1, 3, 10)
#     for mi in mi_grid:
#         for sample_size in SAMPLE_SIZE_GRID:
#             generate_normal_sample(mi, sample_size, dim_count=dim_count, 
    
def generate_normal_benchmark_0(out_dir):
    mi = 0
    depends_dim_count = 0
    dim_count = 5
    samples_count = 100
    benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        samples = []
        for i in range(samples_count):
            start_seed = 42
            samples.append(generate_normal_sample(mi, sample_size, dim_count, depends_dim_count, start_seed + i))
        benchmark[sample_size] = samples

    with open(Path(out_dir) / f"normal_{dim_count}_{depends_dim_count}_{mi}.pkl", "wb") as f:
        pickle.dump(benchmark, f)

def generate_normal_benchmark_1(out_dir):
    mi = 2
    depends_dim_count = 2
    dim_count = 7
    samples_count = 100
    benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        samples = []
        for i in range(samples_count):
            start_seed = 42
            samples.append(generate_normal_sample(mi, sample_size, dim_count, depends_dim_count, start_seed + i))
        benchmark[sample_size] = samples

    with open(Path(out_dir) / f"normal_{dim_count}_{depends_dim_count}_{mi}.pkl", "wb") as f:
        pickle.dump(benchmark, f)

def generate_uniform_benchmark_0(out_dir):
    mi = 2
    depends_dim_count = 2
    dim_count = 9
    samples_count = 100
    benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        samples = []
        for i in range(samples_count):
            start_seed = 42
            samples.append(generate_uniform_sample(mi, sample_size, dim_count, depends_dim_count, start_seed + i))
        benchmark[sample_size] = samples

    with open(Path(out_dir) / f"uniform_{dim_count}_{depends_dim_count}_{mi}.pkl", "wb") as f:
        pickle.dump(benchmark, f)

def generate_loggammaexpon_benchmark_0(out_dir):
    theta = 1
    samples_count = 100
    benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        samples = []
        for i in range(samples_count):
            start_seed = 42
            samples.append(generate_loggammaexpon_sample(theta=theta, sample_size=sample_size, seed=start_seed + i))
        benchmark[sample_size] = samples

    with open(Path(out_dir) / f"loggammaexpon_{theta}.pkl", "wb") as f:
        pickle.dump(benchmark, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сгенерировать и сохранить бенчмарк")
    ap.add_argument("--out", required=True, help="папка для бенчмарка, например bench_v1")
    args = ap.parse_args()

    generate_functions = [
        generate_normal_benchmark_0, generate_normal_benchmark_1,
        generate_uniform_benchmark_0,
        generate_loggammaexpon_benchmark_0
    ]
    os.makedirs(args.out, exist_ok=True)
    for func in generate_functions:
        func(args.out)
    print(f"Сохранено в {args.out}/")