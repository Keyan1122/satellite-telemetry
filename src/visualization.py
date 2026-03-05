import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =================================================
# Utilities
# =================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_seed_dir(seed):
    return os.path.join(get_project_root(), "results", f"seed_{seed}")


def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


# =================================================
# Core Plot Functions
# =================================================
def plot_reconstruction_error(recon_error, labels=None, save_path=None):
    plt.figure(figsize=(14, 4))
    plt.plot(recon_error, linewidth=1)

    if labels is not None:
        anomaly_idx = np.where(labels == 1)[0]
        plt.scatter(anomaly_idx, recon_error[anomaly_idx], s=10)

    plt.xlabel("Time Window Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error Over Time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_uncertainty(uncertainty, save_path=None):
    plt.figure(figsize=(14, 4))
    plt.plot(uncertainty, linewidth=1)

    plt.xlabel("Time Window Index")
    plt.ylabel("Predictive Uncertainty")
    plt.title("Predictive Uncertainty Over Time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_vs_uncertainty(recon_error, uncertainty, labels=None, save_path=None):
    plt.figure(figsize=(6, 6))

    if labels is not None:
        scatter = plt.scatter(
            recon_error,
            uncertainty,
            c=labels,
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(recon_error, uncertainty, alpha=0.6)

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Predictive Uncertainty")
    plt.title("Error vs Uncertainty")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_coverage_risk(
    coverage,
    risk_mean,
    risk_lo,
    risk_hi,
    label,
):
    plt.plot(coverage, risk_mean, label=label)
    plt.fill_between(coverage, risk_lo, risk_hi, alpha=0.2)


# =================================================
# Per-Mode Visualization
# =================================================
def visualize_single_mode(mode_dir):
    figures_dir = os.path.join(mode_dir, "figures")
    ensure_dir(figures_dir)

    recon_error = safe_load(os.path.join(mode_dir, "recon_error.npy"))
    uncertainty = safe_load(os.path.join(mode_dir, "uncertainty.npy"))
    labels = safe_load(os.path.join(mode_dir, "labels.npy"))

    plot_reconstruction_error(
        recon_error,
        labels,
        save_path=os.path.join(figures_dir, "reconstruction_error.png"),
    )

    plot_uncertainty(
        uncertainty,
        save_path=os.path.join(figures_dir, "uncertainty.png"),
    )

    plot_error_vs_uncertainty(
        recon_error,
        uncertainty,
        labels,
        save_path=os.path.join(figures_dir, "error_vs_uncertainty.png"),
    )


# =================================================
# Ablation Plots (Per Seed)
# =================================================
def visualize_ablation(seed_dir):
    mc_dir = os.path.join(seed_dir, "mc_dropout")
    no_dir = os.path.join(seed_dir, "no_dropout")

    cov_mc = safe_load(os.path.join(mc_dir, "coverage.npy"))
    risk_mc = safe_load(os.path.join(mc_dir, "risk_mean.npy"))
    risk_mc_lo = safe_load(os.path.join(mc_dir, "risk_lo.npy"))
    risk_mc_hi = safe_load(os.path.join(mc_dir, "risk_hi.npy"))

    cov_no = safe_load(os.path.join(no_dir, "coverage.npy"))
    risk_no = safe_load(os.path.join(no_dir, "risk_mean.npy"))
    risk_no_lo = safe_load(os.path.join(no_dir, "risk_lo.npy"))
    risk_no_hi = safe_load(os.path.join(no_dir, "risk_hi.npy"))

    # Coverage–Risk Plot
    plt.figure(figsize=(6, 6))
    plot_coverage_risk(cov_mc, risk_mc, risk_mc_lo, risk_mc_hi, "MC Dropout")
    plot_coverage_risk(cov_no, risk_no, risk_no_lo, risk_no_hi, "No Dropout")

    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Coverage–Risk Trade-off")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, "coverage_risk_ablation.png"))
    plt.close()

    # AUROC Bar Plot
    with open(os.path.join(mc_dir, "metrics.json")) as f:
        mc_metrics = json.load(f)

    with open(os.path.join(no_dir, "metrics.json")) as f:
        no_metrics = json.load(f)

    plt.figure(figsize=(4, 4))
    plt.bar(
        ["MC Dropout", "No Dropout"],
        [mc_metrics["AUROC"], no_metrics["AUROC"]],
    )
    plt.ylabel("AUROC")
    plt.title("AUROC Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, "auroc_ablation.png"))
    plt.close()


# =================================================
# Master Entry
# =================================================
def run_visualization(seed):
    seed_dir = get_seed_dir(seed)

    if not os.path.exists(seed_dir):
        print(f"Seed directory not found: {seed_dir}")
        return

    print(f"\nGenerating visualizations for seed {seed}...")

    try:
        visualize_single_mode(os.path.join(seed_dir, "mc_dropout"))
        visualize_single_mode(os.path.join(seed_dir, "no_dropout"))
        visualize_ablation(seed_dir)

        print("✓ Visualization complete.\n")

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
