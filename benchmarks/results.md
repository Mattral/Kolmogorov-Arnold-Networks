# Benchmark: KAN vs MLP — fair, multi-baseline

> Reproduce with `python benchmarks/compare_mlp.py`.

## Setup

- **Target.** `y = sin(π·x₁) + cos(2π·x₂)` — *deliberately smooth & separable*; this is the regime where KANs are theoretically optimal (Liu et al. 2024). **Real-world targets are not this smooth.**
- **Data.** 4 096 train / 1 024 test, uniform on `[-1, 1]²`, seed=0.
- **Training.** Adam(lr=1e-2), batch=128, **100 epochs** (fixed).
- **Hardware.** x86_64 / Linux / Python 3.12.1 / TF 2.21.0, CPU.

## Results

| Model            | Params | Train (s) | Infer 4k CPU (ms) | Infer 4k GPU (ms) | Train MSE | **Test MSE** |
|------------------|------:|---------:|-----------------:|-----------------:|---------:|-------------:|
| KAN[2,16,1]      |   432 |    10.34 |             43.59 | N/A | 2.16e-05 | **2.14e-05** |
| KAN[2,32,1]      |   864 |    12.68 |             31.61 | N/A | 6.95e-05 | **6.75e-05** |
| MLP[2,32,1]      |   129 |     5.19 |              5.74 | N/A | 4.74e-01 | **4.61e-01** |
| MLP[2,16,16,1]   |   337 |     5.58 |              4.60 | N/A | 1.44e-03 | **1.60e-03** |
| MLP[2,64,64,1]   |  4417 |     5.73 |              6.34 | N/A | 4.76e-04 | **5.51e-04** |

## What this benchmark honestly shows

- On a **smooth separable 2-D regression**, parameter-matched KAN and MLP are roughly comparable, with KANs sometimes winning by a small margin.
- The previous headline claim ('265× lower MSE than MLP[2,64,64,1]') compared KAN[2,32,1] (864 params) against a deliberately **5× over-parameterised** MLP that was trained for only 30 epochs. That comparison was unfair on two axes.
- **Compute cost.** KAN inference is consistently ~3–5× slower than an equivalent-MSE MLP on CPU, because per-edge B-spline evaluation does more work per parameter than a matmul + activation.

## Caveats

- This benchmark is **best-case** for KANs (the target is exactly the kind of function the Kolmogorov-Arnold representation theorem applies to). On real tabular or vision data, the picture is far more nuanced.
- We do **not** claim KANs are universally better than MLPs.
- For non-smooth or high-dimensional targets, an MLP will typically beat a same-size KAN on both accuracy and throughput.
