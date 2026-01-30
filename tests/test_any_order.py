"""
Sanity check that UPCA, Tucker (MPCA), CP (TROD) work for tensors of any order (not just 4D images).
Run: python -m tests.test_any_order
"""
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor_monitoring import UPCAControlChart, TuckerControlChart, CPControlChart


def test_3d_tensors():
    """3D: (N, d1, d2)."""
    np.random.seed(42)
    N, d1, d2 = 50, 20, 15
    X = np.random.randn(N, d1, d2).astype(np.float64)

    upca = UPCAControlChart(n_components=5, alpha=0.05)
    upca.fit(X)
    t2_a, q_a, _, _ = upca.predict(X[:5])
    assert t2_a.shape == (5,) and q_a.shape == (5,), "UPCA 3D"

    mpca = TuckerControlChart(rank=(4, 3), alpha=0.05)
    mpca.fit(X)
    t2_a, q_a, _, _ = mpca.predict(X[:5])
    assert t2_a.shape == (5,) and q_a.shape == (5,), "Tucker (MPCA) 3D"

    trod = CPControlChart(rank=4, alpha=0.05)
    trod.fit(X)
    t2_a, q_a, _, _ = trod.predict(X[:5])
    assert t2_a.shape == (5,) and q_a.shape == (5,), "CP (TROD) 3D"

    print("3D (N, d1, d2): UPCA, Tucker (MPCA), CP (TROD) OK.")


def test_5d_tensors():
    """5D: (N, d1, d2, d3, d4)."""
    np.random.seed(42)
    N, d1, d2, d3, d4 = 30, 5, 6, 4, 5
    X = np.random.randn(N, d1, d2, d3, d4).astype(np.float64)

    upca = UPCAControlChart(n_components=4, alpha=0.05)
    upca.fit(X)
    t2_a, q_a, _, _ = upca.predict(X[:3])
    assert t2_a.shape == (3,) and q_a.shape == (3,), "UPCA 5D"

    mpca = TuckerControlChart(rank=(2, 2, 2, 2), alpha=0.05)
    mpca.fit(X)
    t2_a, q_a, _, _ = mpca.predict(X[:3])
    assert t2_a.shape == (3,) and q_a.shape == (3,), "Tucker (MPCA) 5D"

    trod = CPControlChart(rank=3, alpha=0.05)
    trod.fit(X)
    t2_a, q_a, _, _ = trod.predict(X[:3])
    assert t2_a.shape == (3,) and q_a.shape == (3,), "CP (TROD) 5D"

    print("5D (N, d1, d2, d3, d4): UPCA, Tucker (MPCA), CP (TROD) OK.")


if __name__ == "__main__":
    test_3d_tensors()
    test_5d_tensors()
    print("All any-order tests passed.")
