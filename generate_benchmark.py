import argparse
import pickle
import os
import platform
import numpy as np
import pandas as pd

import scipy.stats as sps

"""
Создаём бенчмарки для оценки качества критерия независимости
"""

# список диффеоморфизмов

def non_linear_shift(X, Y):
    a = 2
    b = 1
    return X, Y + a * np.arctan(b * Y)

def non_linear_radial(X, Y):
    alpha = 0.5
    return X * np.sqrt(1 + alpha * (X * X)), Y

BENCH_VERSION = "1.0"

DIFFEOMORPHISMS = [
    non_linear_shift, non_linear_radial
]

SAMPLE_SIZE_GRID = [110, 250, 500, 1000, 2500, 5000, 10000]
DIM_GRID = [2, 5, 10, 20, 50, 100]

def generate_zero_correlation_0(
    depends_pair_count=50,
    undepends_pair_count=50,
):
    # X ~ Лаплас
    # Y = cos(4 * X) + eps, eps ~ N(0, 1)
    result_benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        # создаем benchmark с размером выборок sample_size
        sample_benchmark = []
        depend_mask = np.random.permutation(np.concat([np.ones(depends_pair_count), np.zeros(undepends_pair_count)]))
        for depend_flag in depend_mask:
            curr_dim = np.random.choice(np.array(DIM_GRID))
            X = sps.laplace(0, 2).rvs(size=(sample_size, curr_dim))
            eps = sps.norm(0, 1).rvs(size=(sample_size, curr_dim))
            if depend_flag:
                Y = np.cos(4 * X) + eps
            else:
                new_X = sps.laplace(0, 2).rvs(size=(sample_size, curr_dim))
                Y = np.cos(4 * new_X) + eps
        
            sample_benchmark.append({
                "X": X,
                "Y": Y,
                "label": depend_flag
            })
        
        result_benchmark[sample_size] = sample_benchmark
    
    return result_benchmark

def generate_zero_correlation_1(
    depends_pair_count=50,
    undepends_pair_count=50,
):
    # X ~ N(0, 1)
    # Y = sin(X^2) + eps, eps ~ N(0, 0.5)
    result_benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        # создаем benchmark с размером выборок sample_size
        sample_benchmark = []
        depend_mask = np.random.permutation(np.concat([np.ones(depends_pair_count), np.zeros(undepends_pair_count)]))
        for depend_flag in depend_mask:
            curr_dim = np.random.choice(np.array(DIM_GRID))
            X = sps.norm(0, 1).rvs(size=(sample_size, curr_dim))
            eps = sps.norm(0, 0.5).rvs(size=(sample_size, curr_dim))
            if depend_flag:
                Y = np.sin(X ** 2) + eps
            else:
                new_X = sps.norm(0, 1).rvs(size=(sample_size, curr_dim))
                Y = np.sin(new_X ** 2) + eps

            sample_benchmark.append({
                "X": X,
                "Y": Y,
                "label": depend_flag,
            })

        result_benchmark[sample_size] = sample_benchmark

    return result_benchmark

def generate_zero_correlation_2(
    depends_pair_count=50,
    undepends_pair_count=50,
):
    # X ~ N(0, 4)
    # Y = sign(Z - 1 / 2) * X + eps
    # Z ~ Bern(1 / 2)
    # eps ~ N(0, 1)
    result_benchmark = dict()
    for sample_size in SAMPLE_SIZE_GRID:
        # создаем benchmark с размером выборок sample_size
        sample_benchmark = []
        depend_mask = np.random.permutation(np.concat([np.ones(depends_pair_count), np.zeros(undepends_pair_count)]))
        for depend_flag in depend_mask:
            curr_dim = np.random.choice(np.array(DIM_GRID))
            X = sps.norm(0, 4).rvs(size=(sample_size, curr_dim))
            eps = sps.norm(0, 1).rvs(size=(sample_size, curr_dim))
            Z = np.sign(sps.uniform(-1, 1).rvs(size=(sample_size, curr_dim)))

            if depend_flag:
                Y = Z * X + eps
            else:
                new_X = sps.norm(0, 4).rvs(size=(sample_size, curr_dim))
                Y = Z * new_X + eps

            X_mut, Y_mut = DIFFEOMORPHISMS[0](X, Y)
            sample_benchmark.append({
                "X": X_mut,
                "Y": Y_mut,
                "label": depend_flag,
            })

        result_benchmark[sample_size] = sample_benchmark

    return result_benchmark

def save_benchmark(out_dir, seed=0):
    os.makedirs(os.path.join(out_dir, "cases"), exist_ok=True)
    for gen_func, label in [
        (generate_zero_correlation_0, "zero_correlation_0"),
        (generate_zero_correlation_1, "zero_correlation_1"),
        (generate_zero_correlation_2, "zero_correlation_2")
    ]:
        out_file = os.path.join(out_dir, "cases", label + ".pkl")
        with open(out_file, "wb") as f:
            benchmark = gen_func()
            pickle.dump(benchmark, f)
            print(f"save {label}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сгенерировать и сохранить бенчмарк")
    ap.add_argument("--out", required=True, help="папка для бенчмарка, например bench_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    save_benchmark(args.out, args.seed)
    print(f"Сохранено в {args.out}/")