import os
import json
import numpy as np
import pandas as pd

from statistics import significance_test

RESULTS_DIR = "results"

def compute_risk_at_coverage(seed_dir, mode, target_coverage=0.9):
    cov = np.load(os.path.join(seed_dir, mode, "coverage.npy"))
    risk = np.load(os.path.join(seed_dir, mode, "risk_mean.npy"))
    idx = (np.abs(cov - target_coverage)).argmin()
    return float(risk[idx])


def aggregate_metric(values):
    return np.mean(values), np.std(values)


def generate_table():
    seed_dirs = [
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if d.startswith("seed_")
    ]

    mc_auroc, no_auroc = [], []
    mc_pr, no_pr = [], []
    mc_risk, no_risk = [], []

    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, "mc_dropout", "metrics.json")) as f:
            mc_metrics = json.load(f)
        with open(os.path.join(seed_dir, "no_dropout", "metrics.json")) as f:
            no_metrics = json.load(f)

        mc_auroc.append(mc_metrics["AUROC"])
        no_auroc.append(no_metrics["AUROC"])

        mc_pr.append(mc_metrics["PR_AUC"])
        no_pr.append(no_metrics["PR_AUC"])

        mc_risk.append(compute_risk_at_coverage(seed_dir, "mc_dropout"))
        no_risk.append(compute_risk_at_coverage(seed_dir, "no_dropout"))

    # Aggregate
    mc_auroc_m, mc_auroc_s = aggregate_metric(mc_auroc)
    no_auroc_m, no_auroc_s = aggregate_metric(no_auroc)

    mc_pr_m, mc_pr_s = aggregate_metric(mc_pr)
    no_pr_m, no_pr_s = aggregate_metric(no_pr)

    mc_risk_m, mc_risk_s = aggregate_metric(mc_risk)
    no_risk_m, no_risk_s = aggregate_metric(no_risk)

    # Significance
    stats = significance_test(np.array(mc_auroc), np.array(no_auroc))

    df = pd.DataFrame([
        {
            "Mode": "MC Dropout",
            "AUROC (mean±std)": f"{mc_auroc_m:.3f} ± {mc_auroc_s:.3f}",
            "PR_AUC (mean±std)": f"{mc_pr_m:.3f} ± {mc_pr_s:.3f}",
            "Risk@90% (mean±std)": f"{mc_risk_m:.3f} ± {mc_risk_s:.3f}",
        },
        {
            "Mode": "No Dropout",
            "AUROC (mean±std)": f"{no_auroc_m:.3f} ± {no_auroc_s:.3f}",
            "PR_AUC (mean±std)": f"{no_pr_m:.3f} ± {no_pr_s:.3f}",
            "Risk@90% (mean±std)": f"{no_risk_m:.3f} ± {no_risk_s:.3f}",
        },
        {
            "Mode": "p-value (AUROC)",
            "AUROC (mean±std)": f"{stats['paired_t_test']['p_value']:.4f}",
            "PR_AUC (mean±std)": "",
            "Risk@90% (mean±std)": "",
        },
        {
            "Mode": "Cohen's d",
            "AUROC (mean±std)": f"{stats['effect_size']['cohens_d']:.3f} "
                                f"({stats['effect_size']['interpretation']})",
            "PR_AUC (mean±std)": "",
            "Risk@90% (mean±std)": "",
        }
    ])

    csv_path = os.path.join(RESULTS_DIR, "results_table.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(RESULTS_DIR, "results_table.md")
    with open(md_path, "w") as f:
        f.write(df.to_string(index=False))

    print("\nMulti-seed results table:\n")
    print(df)
    print(f"\nSaved to:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    generate_table()
