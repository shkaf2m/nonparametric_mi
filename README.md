Процесс оценки качества непараметрических оценщиков взаимной информации (CCA, KSG, WKL):
1) Генерация бенчмарков: python3 generate_benchmark_validate.py --out benchmarks
2) Запуск оценщиков: python3 run_estimators_validate.py --benchmarks benchmarks --out results
3) Визуализация графиков: python3 visualize.py --results results
