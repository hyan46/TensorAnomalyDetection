"""
Paper simulation: symmetric/asymmetric images, location and area shifts.
Computes F1, precision, recall; plots T² and Q control charts.
Run from repo root: python -m tests.test_monitoring_paper
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure package root is on path when run as script or -m
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor_monitoring import UPCAControlChart, TuckerControlChart, CPControlChart
from sklearn.metrics import f1_score, precision_score, recall_score


def generate_symmetric_image(size=(50, 50), random_state=None, shift_x=0, shift_y=0, radius_factor=1.0):
    scale = size[0] / 200.0
    center_mean = np.array([100 + shift_x, 100 + shift_y]) * scale
    center_cov = np.eye(2) * (2 * scale) ** 2
    if random_state is not None:
        np.random.seed(random_state)
    center = np.random.multivariate_normal(center_mean, center_cov)
    radius = 25 * scale * radius_factor
    color = np.random.normal([200, 50, 50], [5, 5, 5])
    y, x = np.ogrid[: size[0], : size[1]]
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2
    img = np.zeros((size[0], size[1], 3))
    for c in range(3):
        img[:, :, c][mask] = color[c]
    img += np.random.normal(0, 2, img.shape)
    return np.clip(img, 0, 255)


def generate_asymmetric_image(
    size=(50, 50), random_state=None, shift_x=0, shift_y=0, radius_factor=1.0, noise_std=3.5
):
    """Two overlapping circles (red + green). Harder than symmetric: more noise, subtler changes."""
    scale = size[0] / 200.0
    if random_state is not None:
        np.random.seed(random_state)
    img = np.zeros((size[0], size[1], 3))
    c1_center = np.random.multivariate_normal(
        np.array([80 + shift_x, 80 + shift_y]) * scale, np.eye(2) * (2 * scale) ** 2
    )
    c1_radius = 25 * scale * radius_factor
    c1_color = np.random.normal([200, 50, 50], [6, 6, 6])
    c2_center = np.random.multivariate_normal(
        np.array([120 + shift_x, 120 + shift_y]) * scale, np.eye(2) * (2 * scale) ** 2
    )
    c2_radius = 35 * scale * radius_factor
    c2_color = np.random.normal([50, 200, 50], [6, 6, 6])
    y, x = np.ogrid[: size[0], : size[1]]
    mask1 = ((x - c1_center[0]) ** 2 + (y - c1_center[1]) ** 2) <= c1_radius ** 2
    mask2 = ((x - c2_center[0]) ** 2 + (y - c2_center[1]) ** 2) <= c2_radius ** 2
    for c in range(3):
        img[:, :, c][mask2] = c2_color[c]
        img[:, :, c][mask1] = c1_color[c]
    img += np.random.normal(0, noise_std, img.shape)
    return np.clip(img, 0, 255)


def _plot_sample_images(images, title, filename):
    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].astype(int))
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()


def _plot_control_charts(t2_stats, q_stats, ucl_t2, ucl_q, title, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t2_stats, "b-", label="T²")
    plt.axhline(ucl_t2, color="r", linestyle="--", label="UCL")
    plt.title(f"{title} - T² Chart")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(q_stats, "g-", label="Q")
    plt.axhline(ucl_q, color="r", linestyle="--", label="UCL")
    plt.title(f"{title} - Q Chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def _safe_filename(name):
    """Turn display name into filename-safe string, e.g. 'Tucker (MPCA)' -> 'Tucker_MPCA'."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _evaluate_model(model, X_test, y_true, model_name, scenario_name, output_dir):
    t0 = time.time()
    t2_alarm, q_alarm, t2_stats, q_stats = model.predict(X_test)
    elapsed = time.time() - t0
    any_alarm = t2_alarm | q_alarm
    safe_name = _safe_filename(model_name)
    plot_path = os.path.join(output_dir, f"{scenario_name}_{safe_name}_chart.png")
    _plot_control_charts(t2_stats, q_stats, model.ucl_t2_, model.ucl_q_, f"{model_name} on {scenario_name}", plot_path)
    return {
        "F1": f1_score(y_true, any_alarm, zero_division=0),
        "Precision": precision_score(y_true, any_alarm, zero_division=0),
        "Recall": recall_score(y_true, any_alarm, zero_division=0),
        "Time": elapsed,
    }


def run_simulation(write_report=True):
    output_dir = os.path.join(_ROOT, "monitoring_results")
    os.makedirs(output_dir, exist_ok=True)

    n_train = 200
    n_test_normal = 50
    n_test_anomaly = 50
    img_size = (32, 32)
    alpha = 0.005

    all_results = {}

    # Asymmetric: harder — subtler shift/area and more color noise
    shift_sym, shift_asym = 20, 8
    area_sym, area_asym = 1.2, 1.08

    for scenario in ("Symmetric", "Asymmetric"):
        print(f"\n=== Running Scenario: {scenario} ===")
        is_asym = scenario == "Asymmetric"
        gen = generate_asymmetric_image if is_asym else generate_symmetric_image
        shift_x, shift_y = (shift_asym, shift_asym) if is_asym else (shift_sym, shift_sym)
        area_factor = area_asym if is_asym else area_sym

        X_train = [gen(img_size) for _ in range(n_train)]
        X_test_normal = [gen(img_size) for _ in range(n_test_normal)]
        X_test_loc = [gen(img_size, shift_x=shift_x, shift_y=shift_y) for _ in range(n_test_anomaly)]
        X_test_area = [gen(img_size, radius_factor=area_factor) for _ in range(n_test_anomaly)]

        _plot_sample_images(X_train, f"{scenario} Training", os.path.join(output_dir, f"{scenario}_train.png"))
        _plot_sample_images(X_test_loc, f"{scenario} Location Shift", os.path.join(output_dir, f"{scenario}_loc.png"))
        _plot_sample_images(X_test_area, f"{scenario} Area Change", os.path.join(output_dir, f"{scenario}_area.png"))

        X_test_1 = X_test_normal + X_test_loc
        y_test_1 = np.array([0] * n_test_normal + [1] * n_test_anomaly)
        X_test_2 = X_test_normal + X_test_area
        y_test_2 = np.array([0] * n_test_normal + [1] * n_test_anomaly)

        models = {
            "UPCA": UPCAControlChart(n_components=5, alpha=alpha),
            "Tucker (MPCA)": TuckerControlChart(rank=(4, 4, 3), alpha=alpha),
            "CP (TROD)": CPControlChart(rank=5, alpha=alpha),
        }

        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train)
            results[name] = {
                "Location": _evaluate_model(model, X_test_1, y_test_1, name, f"{scenario}_Location", output_dir),
                "Area": _evaluate_model(model, X_test_2, y_test_2, name, f"{scenario}_Area", output_dir),
            }
        all_results[scenario] = results

        print(f"\nResults for {scenario}:")
        print(f"{'Model':<10} | {'Test':<10} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Time (s)':<8}")
        print("-" * 60)
        for name, res in results.items():
            for test_type in ("Location", "Area"):
                m = res[test_type]
                print(f"{name:<10} | {test_type:<10} | {m['F1']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['Time']:.4f}")

    if write_report:
        report_path = os.path.join(output_dir, "f1_report.txt")
        lines = [
            "F1 Score Report — Tensor Monitoring (Paper Simulation)",
            "=" * 55,
            f"Settings: n_train={n_train}, n_test_normal={n_test_normal}, n_test_anomaly={n_test_anomaly}, img_size={img_size}, alpha={alpha}",
            "",
        ]
        for scenario, results in all_results.items():
            lines.append(f"Scenario: {scenario}")
            lines.append("-" * 40)
            for name, res in results.items():
                for test_type in ("Location", "Area"):
                    m = res[test_type]
                    lines.append(f"  {name} / {test_type}:  F1={m['F1']:.4f}  Precision={m['Precision']:.4f}  Recall={m['Recall']:.4f}")
            lines.append("")
        lines.append("(Alarm = T² > UCL or Q > UCL; label 1 = anomaly.)")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nF1 report written to: {report_path}")

    return all_results


if __name__ == "__main__":
    run_simulation()
