# Image-Based Process Monitoring Using Low-Rank Tensor Decomposition

Reference implementation and reproduction materials for the paper:

> **H. Yan, K. Paynabar, and J. Shi**, “Image-based process monitoring using low-rank tensor decomposition,” *IEEE Trans. Autom. Sci. Eng.*, vol. 12, no. 1, pp. 216–227, Jan. 2015.  
> **DOI:** [10.1109/TASE.2014.2327029](https://doi.org/10.1109/TASE.2014.2327029)

---

## Citation

```bibtex
@article{yan2015image,
  title={Image-based process monitoring using low-rank tensor decomposition},
  author={Yan, Hao and Paynabar, Kamran and Shi, Jianjun},
  journal={IEEE Transactions on Automation Science and Engineering},
  volume={12},
  number={1},
  pages={216--227},
  year={2015},
  publisher={IEEE},
  doi={10.1109/TASE.2014.2327029}
}
```

---

## Abstract

Image and video sensors are increasingly used in process monitoring. Existing techniques often fail to fully use the information in color images due to high dimensionality and complex correlation (temporal, spatial, spectral). This work models image data as tensors and uses low-rank tensor decomposition (UPCA, MPCA, TROD) to extract monitoring features, then applies multivariate control charts (T² and Q). The paper establishes relationships between Tucker/MPCA and CP/TROD and compares methods on symmetric/asymmetric image simulations and a steel tube flame monitoring case study.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `scikit-learn`, `tensorly`, `matplotlib`.

---

## Methods Implemented

| Method | Description |
|--------|-------------|
| **UPCA** | Unfolded PCA: unfold tensor to vector, PCA, then T² and Q charts. |
| **MPCA** | Multilinear PCA (Tucker): projection per mode; monitor core features and residuals. |
| **TROD** | Tensor rank-one decomposition (CP): shared rank-one factors; monitor weights and residuals. |

All support tensors of shape `(N, d1, d2, ..., d_m)` (e.g. `(N, Height, Width, Channels)` for RGB images). Control limits use empirical \((1-\alpha)\) percentiles; \(\alpha \approx 0.005\) targets in-control ARL ≈ 200.

### Quick usage

```python
from tensor_monitoring import UPCAControlChart, MPCAControlChart, TRODControlChart

# Example: UPCA
model = UPCAControlChart(n_components=5, alpha=0.005)
model.fit(X_train)  # (N, H, W, C)
t2_alarm, q_alarm, t2_stats, q_stats = model.predict(X_test)

# MPCA (rank per mode)
model = MPCAControlChart(rank=(4, 4, 3), alpha=0.005)
model.fit(X_train)

# TROD (number of rank-one components)
model = TRODControlChart(rank=5, alpha=0.005)
model.fit(X_train)
```

---

## Reproducing Results

Run the paper-style simulation (symmetric/asymmetric images, location and area shifts, T² and Q control charts):

```bash
python -m tests.test_monitoring_paper
```

Outputs are written to `monitoring_results/`.

---

## Results (from `monitoring_results/`)

Initial reproduction results follow the paper’s setup: symmetric (single circle) and asymmetric (two overlapping circles) RGB images, with in-control training and out-of-control tests for **location shift** and **area change**. Metrics (F1, precision, recall) and control charts are below.

### Sample images

| Scenario    | Training | Location shift | Area change |
|------------|----------|----------------|-------------|
| Symmetric  | [`Symmetric_train.png`](monitoring_results/Symmetric_train.png) | [`Symmetric_loc.png`](monitoring_results/Symmetric_loc.png) | [`Symmetric_area.png`](monitoring_results/Symmetric_area.png) |
| Asymmetric | [`Asymmetric_train.png`](monitoring_results/Asymmetric_train.png) | [`Asymmetric_loc.png`](monitoring_results/Asymmetric_loc.png) | [`Asymmetric_area.png`](monitoring_results/Asymmetric_area.png) |

### Detection performance (F1 / Precision / Recall)

From [`monitoring_results/f1_report.txt`](monitoring_results/f1_report.txt) (alarm = T² > UCL or Q > UCL):

**Symmetric**

| Method | Location       | Area           |
|--------|----------------|----------------|
| UPCA   | 0.99 / 0.98 / 1.00 | 0.99 / 0.98 / 1.00 |
| MPCA   | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| TROD   | 0.99 / 0.98 / 1.00 | 0.99 / 0.98 / 1.00 |

**Asymmetric**

| Method | Location       | Area           |
|--------|----------------|----------------|
| UPCA   | 1.00 / 1.00 / 1.00 | 0.97 / 1.00 / 0.94 |
| MPCA   | 1.00 / 1.00 / 1.00 | **0.99 / 1.00 / 0.98** |
| TROD   | 1.00 / 1.00 / 1.00 | 0.36 / 1.00 / 0.22 |

Consistent with the paper: **MPCA** performs best on asymmetric (location/color interaction) scenarios; TROD excels when the foreground is more symmetric.

### Control charts (T² and Q)

Example charts for **Asymmetric / Location** and **Asymmetric / Area**:

- [Asymmetric_Location_UPCA_chart.png](monitoring_results/Asymmetric_Location_UPCA_chart.png)
- [Asymmetric_Location_MPCA_chart.png](monitoring_results/Asymmetric_Location_MPCA_chart.png)
- [Asymmetric_Location_TROD_chart.png](monitoring_results/Asymmetric_Location_TROD_chart.png)
- [Asymmetric_Area_UPCA_chart.png](monitoring_results/Asymmetric_Area_UPCA_chart.png)
- [Asymmetric_Area_MPCA_chart.png](monitoring_results/Asymmetric_Area_MPCA_chart.png)
- [Asymmetric_Area_TROD_chart.png](monitoring_results/Asymmetric_Area_TROD_chart.png)

Symmetric counterparts are in the same folder (e.g. `Symmetric_Location_UPCA_chart.png`, `Symmetric_Area_MPCA_chart.png`).

---

## Project layout

```
├── README.md
├── CITATION.bib
├── requirements.txt
├── LICENSE
├── .gitignore
├── __init__.py
├── tensor_monitoring.py      # UPCA, MPCA, TROD (paper methods)
├── tensor_decomposition_utils.py
├── tensor_anomaly_detection.py
├── tensordetect.py
├── three_way_decomposition.py
├── monitoring_results/       # Sample images, control charts, f1_report.txt
└── tests/
    ├── __init__.py
    ├── test_monitoring_paper.py
    └── test_any_order.py
```

---

## License

MIT License.
