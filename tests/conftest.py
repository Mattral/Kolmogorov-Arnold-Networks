"""Pytest configuration: add `src/` to sys.path so `import kanx` works.

We also lock thread counts for both TF and (optionally) PyTorch to 1 thread.
Mixing TF + Torch in a single process is known to segfault under contention
when both libraries spawn their own oneDNN / OpenMP / sleef thread pools.
Single-threaded BLAS keeps the test suite deterministic and hermetic at
the cost of marginal wall-clock time.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Threading hygiene — must be set before any C extension is loaded.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# If torch is installed, single-thread it too.
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
