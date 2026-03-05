import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
from src.statistics import significance_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
AGG_DIR = os.path.join(RESULTS_DIR, "aggregated")

os.makedirs(AGG_DIR, exist_ok=True)


def get_seed_dirs():
    return [
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if d.startswith("seed_")
    ]


def mean_ci(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    h = std * t.ppf((1 + confidence) / 2., n-1) / np.sqrt(n)
    return mean, std, h


def aggregate_auroc(seed_dirs):
    mc_vals, no_vals = [], []

    for seed in seed_dirs:
        with open(os.path.join(seed, "mc_dropout", "metrics.json")) as f:
            mc = json.load(f)
        with open(os.path.join(seed, "no_dropout", "metrics.json")) as f:
            no = json.load(f)

        mc_vals.append(mc["AUROC"])
        no_vals.append(no["AUROC"])

    return mc_vals, no_vals


def aggregate_coverage_risk(seed_dirs):
    mc_risks = []
    no_risks = []
    coverage = None

    for seed in seed_dirs:
        mc_dir = os.path.join(seed, "mc_dropout")
        no_dir = os.path.join(seed, "no_dropout")

        cov = np.load(os.path.join(mc_dir, "coverage.npy"))
        risk_mc = np.load(os.path.join(mc_dir, "risk_mean.npy"))
        risk_no = np.load(os.path.join(no_dir, "risk_mean.npy"))

        coverage = cov
        mc_risks.append(risk_mc)
        no_risks.append(risk_no)

    return coverage, np.array(mc_risks), np.array(no_risks)


def aggregate_threshold_metrics(seed_dirs):
    """
    Load threshold metrics from each seed and return aggregated data.
    Returns: {
        'mc_dropout': {'percentile': {'precision': [], 'recall': [], 'f1': []}, ...},
        'no_dropout': {...}
    }
    """
    mc_data = {method: {"precision": [], "recall": [], "f1": [], "false_alarm_rate": [], "detection_delay_mean": []}
               for method in ["percentile", "roc_optimal", "risk_based"]}
    no_data = {method: {"precision": [], "recall": [], "f1": [], "false_alarm_rate": [], "detection_delay_mean": []}
               for method in ["percentile", "roc_optimal", "risk_based"]}

    for seed in seed_dirs:
        # MC Dropout
        with open(os.path.join(seed, "mc_dropout", "metrics.json")) as f:
            mc_metrics = json.load(f)
        for method, vals in mc_metrics["threshold_metrics"].items():
            for metric in ["precision", "recall", "f1", "false_alarm_rate"]:
                mc_data[method][metric].append(vals[metric])
            # detection delay may be None for some seeds if no events detected; skip None
            delay = vals.get("detection_delay_mean")
            if delay is not None:
                mc_data[method]["detection_delay_mean"].append(delay)

        # No Dropout
        with open(os.path.join(seed, "no_dropout", "metrics.json")) as f:
            no_metrics = json.load(f)
        for method, vals in no_metrics["threshold_metrics"].items():
            for metric in ["precision", "recall", "f1", "false_alarm_rate"]:
                no_data[method][metric].append(vals[metric])
            delay = vals.get("detection_delay_mean")
            if delay is not None:
                mc_data[method]["detection_delay_mean"].append(delay)

    return mc_data, no_data


def aggregate_precision_recall_curves(seed_dirs):
    """
    Aggregate precision and recall vs coverage curves.
    Returns:
        coverage (common x-axis),
        mc_prec_mean, mc_prec_std, no_prec_mean, no_prec_std,
        mc_rec_mean, mc_rec_std, no_rec_mean, no_rec_std
    """
    mc_prec_list, mc_rec_list = [], []
    no_prec_list, no_rec_list = [], []
    coverage = None

    for seed in seed_dirs:
        mc_dir = os.path.join(seed, "mc_dropout")
        no_dir = os.path.join(seed, "no_dropout")

        cov = np.load(os.path.join(mc_dir, "coverage_pr.npy"))   # same for both modes
        coverage = cov   # should be identical across seeds; we take last

        mc_prec = np.load(os.path.join(mc_dir, "precision_vs_coverage.npy"))
        mc_rec  = np.load(os.path.join(mc_dir, "recall_vs_coverage.npy"))
        no_prec = np.load(os.path.join(no_dir, "precision_vs_coverage.npy"))
        no_rec  = np.load(os.path.join(no_dir, "recall_vs_coverage.npy"))

        mc_prec_list.append(mc_prec)
        mc_rec_list.append(mc_rec)
        no_prec_list.append(no_prec)
        no_rec_list.append(no_rec)

    mc_prec_arr = np.array(mc_prec_list)
    mc_rec_arr  = np.array(mc_rec_list)
    no_prec_arr = np.array(no_prec_list)
    no_rec_arr  = np.array(no_rec_list)

    return (coverage,
            mc_prec_arr.mean(axis=0), mc_prec_arr.std(axis=0),
            no_prec_arr.mean(axis=0), no_prec_arr.std(axis=0),
            mc_rec_arr.mean(axis=0),  mc_rec_arr.std(axis=0),
            no_rec_arr.mean(axis=0),  no_rec_arr.std(axis=0))


def plot_aggregated_coverage_risk(seed_dirs):
    coverage, mc_risks, no_risks = aggregate_coverage_risk(seed_dirs)

    mc_mean = mc_risks.mean(axis=0)
    mc_std = mc_risks.std(axis=0)

    no_mean = no_risks.mean(axis=0)
    no_std = no_risks.std(axis=0)

    plt.figure(figsize=(6,6))

    plt.plot(coverage, mc_mean, label="MC Dropout")
    plt.fill_between(coverage, mc_mean-mc_std, mc_mean+mc_std, alpha=0.2)

    plt.plot(coverage, no_mean, label="No Dropout")
    plt.fill_between(coverage, no_mean-no_std, no_mean+no_std, alpha=0.2)

    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Aggregated Coverage–Risk (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "coverage_risk_mean.png"))
    plt.close()


def plot_aggregated_threshold_metrics(mc_data, no_data):
    """
    Bar plots comparing threshold metrics (precision, recall, F1, false alarm rate)
    for both methods, arranged in a 2x2 grid.
    """
    methods = ["percentile", "roc_optimal", "risk_based"]
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
    metrics = ["precision", "recall", "f1", "false_alarm_rate"]
    titles = ["Precision", "Recall", "F1 Score", "False Alarm Rate"]

    # Flatten axes for easy indexing
    axes_flat = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes_flat[i]
        mc_means = []
        mc_cis   = []
        no_means = []
        no_cis   = []
        for m in methods:
            mc_m, _, mc_ci = mean_ci(mc_data[m][metric])
            no_m, _, no_ci = mean_ci(no_data[m][metric])
            mc_means.append(mc_m)
            mc_cis.append(mc_ci)
            no_means.append(no_m)
            no_cis.append(no_ci)

        ax.bar(x - width/2, mc_means, width, yerr=mc_cis, capsize=5, label='MC Dropout')
        ax.bar(x + width/2, no_means, width, yerr=no_cis, capsize=5, label='No Dropout')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "threshold_metrics_comparison.png"))
    plt.close()


def plot_aggregated_precision_recall_curves(
    coverage,
    mc_prec_mean, mc_prec_std,
    no_prec_mean, no_prec_std,
    mc_rec_mean,  mc_rec_std,
    no_rec_mean,  no_rec_std
):
    """
    Plot precision vs coverage and recall vs coverage with std bands.
    """

    # Determine the minimum length among all relevant arrays
    # This is a workaround to handle potential shape mismatches from data generation.
    min_len = min(len(coverage), len(mc_prec_mean), len(no_prec_mean),
                  len(mc_rec_mean), len(no_rec_mean))

    # Truncate all arrays to the minimum length to ensure compatibility for plotting
    coverage = coverage[:min_len]
    mc_prec_mean = mc_prec_mean[:min_len]
    mc_prec_std = mc_prec_std[:min_len]
    no_prec_mean = no_prec_mean[:min_len]
    no_prec_std = no_prec_std[:min_len]
    mc_rec_mean = mc_rec_mean[:min_len]
    mc_rec_std = mc_rec_std[:min_len]
    no_rec_mean = no_rec_mean[:min_len]
    no_rec_std = no_rec_std[:min_len]

    # Precision vs Coverage
    plt.figure(figsize=(6,5))
    plt.plot(coverage, mc_prec_mean, label='MC Dropout')
    plt.fill_between(coverage, mc_prec_mean - mc_prec_std, mc_prec_mean + mc_prec_std, alpha=0.2)
    plt.plot(coverage, no_prec_mean, label='No Dropout')
    plt.fill_between(coverage, no_prec_mean - no_prec_std, no_prec_mean + no_prec_std, alpha=0.2)
    plt.xlabel("Coverage")
    plt.ylabel("Precision")
    plt.title("Precision vs Coverage (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "precision_vs_coverage.png"))
    plt.close()

    # Recall vs Coverage
    plt.figure(figsize=(6,5))
    plt.plot(coverage, mc_rec_mean, label='MC Dropout')
    plt.fill_between(coverage, mc_rec_mean - mc_rec_std, mc_rec_mean + mc_rec_std, alpha=0.2)
    plt.plot(coverage, no_rec_mean, label='No Dropout')
    plt.fill_between(coverage, no_rec_mean - no_rec_std, no_rec_mean + no_rec_std, alpha=0.2)
    plt.xlabel("Coverage")
    plt.ylabel("Recall")
    plt.title("Recall vs Coverage (Mean ± Std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "recall_vs_coverage.png"))
    plt.close()


def plot_aggregated_auroc(seed_dirs):
    mc_vals, no_vals = aggregate_auroc(seed_dirs)

    mc_mean, mc_std, mc_ci = mean_ci(mc_vals)
    no_mean, no_std, no_ci = mean_ci(no_vals)

    plt.figure(figsize=(4,4))
    labels = ["MC Dropout", "No Dropout"]
    means = [mc_mean, no_mean]
    errors = [mc_ci, no_ci]

    plt.bar(labels, means, yerr=errors, capsize=5)
    plt.ylabel("AUROC")
    plt.title("Mean AUROC ± 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "auroc_mean_ci.png"))
    plt.close()

    return mc_vals, no_vals


def generate_summary_table(mc_vals, no_vals):
    mc_mean, mc_std, mc_ci = mean_ci(mc_vals)
    no_mean, no_std, no_ci = mean_ci(no_vals)

    stats = significance_test(np.array(mc_vals), np.array(no_vals))

    df = pd.DataFrame([
        {
            "Metric": "AUROC (MC Dropout)",
            "Mean": mc_mean,
            "Std": mc_std,
            "95% CI": mc_ci,
        },
        {
            "Metric": "AUROC (No Dropout)",
            "Mean": no_mean,
            "Std": no_std,
            "95% CI": no_ci,
        },
        {
            "Metric": "p-value (paired t-test)",
            "Mean": stats["paired_t_test"]["p_value"],
            "Std": "",
            "95% CI": "",
        },
        {
            "Metric": "Cohen's d",
            "Mean": stats["effect_size"]["cohens_d"],
            "Std": "",
            "95% CI": "",
        }
    ])

    df.to_csv(os.path.join(AGG_DIR, "summary_table.csv"), index=False)

    with open(os.path.join(AGG_DIR, "summary_table.md"), "w") as f:
        f.write(df.to_string(index=False))


def generate_threshold_summary_table(mc_data, no_data):
    """
    Create a table of threshold metrics (mean ± CI) for each method and metric.
    """
    rows = []
    for method in ["percentile", "roc_optimal", "risk_based"]:
        for metric in ["precision", "recall", "f1", "false_alarm_rate"]:
            mc_m, _, mc_ci = mean_ci(mc_data[method][metric])
            no_m, _, no_ci = mean_ci(no_data[method][metric])
            rows.append({
                "Method": method,
                "Metric": metric,
                "MC Dropout (mean±CI)": f"{mc_m:.3f}±{mc_ci:.3f}",
                "No Dropout (mean±CI)": f"{no_m:.3f}±{no_ci:.3f}",
            })
        # Detection delay (may have fewer seeds)
        if mc_data[method]["detection_delay_mean"]:
            mc_delay, _, mc_delay_ci = mean_ci(mc_data[method]["detection_delay_mean"])
            delay_str_mc = f"{mc_delay:.2f}±{mc_delay_ci:.2f}"
        else:
            delay_str_mc = "N/A"
        if no_data[method]["detection_delay_mean"]:
            no_delay, _, no_delay_ci = mean_ci(no_data[method]["detection_delay_mean"])
            delay_str_no = f"{no_delay:.2f}±{no_delay_ci:.2f}"
        else:
            delay_str_no = "N/A"
        rows.append({
            "Method": method,
            "Metric": "detection_delay",
            "MC Dropout (mean±CI)": delay_str_mc,
            "No Dropout (mean±CI)": delay_str_no,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AGG_DIR, "threshold_metrics.csv"), index=False)
    with open(os.path.join(AGG_DIR, "threshold_metrics.md"), "w") as f:
        f.write(df.to_string(index=False))


def generate_pr_curves_summary(
    coverage,
    mc_prec_mean, mc_prec_std,
    no_prec_mean, no_prec_std,
    mc_rec_mean,  mc_rec_std,
    no_rec_mean,  no_rec_std
):
    """
    Optionally, pick a specific coverage (e.g., 0.9) and report precision/recall.
    """
    target = 0.9
    idx = np.argmin(np.abs(coverage - target))
    actual_cov = coverage[idx]

    summary = {
        "target_coverage": target,
        "actual_coverage": actual_cov,
        "MC Dropout": {
            "precision": f"{mc_prec_mean[idx]:.3f}±{mc_prec_std[idx]:.3f}",
            "recall":    f"{mc_rec_mean[idx]:.3f}±{mc_rec_std[idx]:.3f}",
        },
        "No Dropout": {
            "precision": f"{no_prec_mean[idx]:.3f}±{no_prec_std[idx]:.3f}",
            "recall":    f"{no_rec_mean[idx]:.3f}±{no_rec_std[idx]:.3f}",
        }
    }

    # Save as JSON for easy reading
    with open(os.path.join(AGG_DIR, "pr_at_90_coverage.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Also create a short table
    df = pd.DataFrame([
        {"Coverage": actual_cov, "Mode": "MC Dropout", "Precision": summary["MC Dropout"]["precision"], "Recall": summary["MC Dropout"]["recall"]},
        {"Coverage": actual_cov, "Mode": "No Dropout", "Precision": summary["No Dropout"]["precision"], "Recall": summary["No Dropout"]["recall"]}
    ])
    df.to_csv(os.path.join(AGG_DIR, "pr_at_90_coverage.csv"), index=False)


def run_aggregation():
    seed_dirs = get_seed_dirs()

    if not seed_dirs:
        print("No seed directories found.")
        return

    print(f"Aggregating {len(seed_dirs)} seeds...")

    # Existing aggregations
    plot_aggregated_coverage_risk(seed_dirs)
    mc_vals, no_vals = plot_aggregated_auroc(seed_dirs)
    generate_summary_table(mc_vals, no_vals)

    # New threshold metrics aggregation
    mc_data, no_data = aggregate_threshold_metrics(seed_dirs)
    plot_aggregated_threshold_metrics(mc_data, no_data)
    generate_threshold_summary_table(mc_data, no_data)

    # New precision/recall vs coverage curves
    (coverage,
     mc_prec_mean, mc_prec_std,
     no_prec_mean, no_prec_std,
     mc_rec_mean,  mc_rec_std,
     no_rec_mean,  no_rec_std) = aggregate_precision_recall_curves(seed_dirs)

    plot_aggregated_precision_recall_curves(
        coverage,
        mc_prec_mean, mc_prec_std,
        no_prec_mean, no_prec_std,
        mc_rec_mean,  mc_rec_std,
        no_rec_mean,  no_rec_std
    )

    generate_pr_curves_summary(
        coverage,
        mc_prec_mean, mc_prec_std,
        no_prec_mean, no_prec_std,
        mc_rec_mean,  mc_rec_std,
        no_rec_mean,  no_rec_std
    )

    print("Aggregation complete.")
    print(f"Saved in: {AGG_DIR}")
