import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from .mc_dropout import enable_mc_dropout
from .utils import set_seed
from .metrics import (
    percentile_threshold,
    roc_optimal_threshold,
    risk_based_threshold,
    compute_auroc,
    compute_pr,
    bootstrap_coverage_risk,
    false_alarm_rate,
    detection_delay,
)


# -------------------------------------------------
# Helper: point-level → window-level labels
# -------------------------------------------------
def point_to_window_labels(point_labels, window_size, stride=1):
    """
    Convert point-wise labels to window-wise labels.
    A window is anomalous if ANY timestep inside it is anomalous.
    """
    window_labels = []
    for start in range(0, len(point_labels) - window_size + 1, stride):
        end = start + window_size
        window_labels.append(int(point_labels[start:end].max()))
    return np.array(window_labels)


# -------------------------------------------------
# Helper: metrics at a given threshold
# -------------------------------------------------
def compute_metrics_at_threshold(scores, labels, thresh):
    preds = (scores > thresh).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    far = false_alarm_rate(labels, preds)
    mean_delay, missed, total = detection_delay(labels, preds)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "false_alarm_rate": float(far),
        "detection_delay_mean": float(mean_delay) if not np.isnan(mean_delay) else None,
        "missed_events": int(missed),
        "total_events": int(total),
    }


# -------------------------------------------------
# Main evaluation function
# -------------------------------------------------
def evaluate(
    model,
    test_dataset,
    labels,
    device="cpu",
    save_dir="results",
    use_mc_dropout=True,
    seed = 42,
    mc_samples = 50,
):
    """
    Evaluate trained LSTM autoencoder using window-by-window inference.

    Args:
        model: Trained PyTorch model.
        test_dataset: TelemetryDataset for testing.
        labels: Ground truth point-wise labels (full test sequence).
        device: 'cpu' or 'cuda'.
        save_dir: Root results directory.
        use_mc_dropout: If True, run MC Dropout inference.
        seed: Random seed (for reproducibility).
        mc_samples: Number of MC forward passes (if use_mc_dropout=True).

    Returns:
        recon_error_np, uncertainty_np, window_labels
    """
    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    set_seed(seed)

    mode = "mc_dropout" if use_mc_dropout else "no_dropout"
    mode_dir = os.path.join(save_dir, f"seed_{seed}", mode)
    os.makedirs(mode_dir, exist_ok=True)

    model.to(device)
    print(f"\nEvaluating seed {seed} – mode: {mode} on device: {device}")

    # -------------------------------------------------
    # Load trained weights
    # -------------------------------------------------
    checkpoint_path = os.path.join("checkpoints", f"model_seed_{seed}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if use_mc_dropout:
        enable_mc_dropout(model)

    # -------------------------------------------------
    # Window-level inference loop
    # -------------------------------------------------
    recon_errors = []
    uncertainties = []
    n_windows = len(test_dataset)

    with torch.no_grad():
        for i in tqdm(range(n_windows), desc=f"Inference ({mode})"):
            x = test_dataset[i].unsqueeze(0).to(device)  # (1, seq_len, features)

            if use_mc_dropout:
                # MC Dropout: multiple forward passes
                recons = []
                for _ in range(mc_samples):
                    recon = model(x)
                    recons.append(recon.unsqueeze(0))
                recons = torch.cat(recons, dim=0)          # (mc_samples, 1, seq_len, features)

                mean_recon = recons.mean(dim=0)            # (1, seq_len, features)
                var_recon = recons.var(dim=0)              # (1, seq_len, features)
                uncertainty = var_recon.mean().item()       # scalar
            else:
                mean_recon = model(x)
                uncertainty = 0.0

            recon_error = ((x - mean_recon) ** 2).mean().item()
            recon_errors.append(recon_error)
            uncertainties.append(uncertainty)

    recon_error_np = np.array(recon_errors)
    uncertainty_np = np.array(uncertainties)

    # -------------------------------------------------
    # Align labels (point-wise → window-wise)
    # -------------------------------------------------
    window_labels = point_to_window_labels(
        labels,
        window_size = test_dataset.window_size,
        stride = test_dataset.stride,
    )
    if len(window_labels) != len(recon_error_np):
        raise ValueError("Label length mismatch with reconstruction windows.")

    # -------------------------------------------------
    # Compute thresholds
    # -------------------------------------------------
    thresh_percentile = percentile_threshold(recon_error_np)
    thresh_roc = roc_optimal_threshold(recon_error_np, window_labels)
    thresh_risk = risk_based_threshold(recon_error_np, uncertainty_np)

    # -------------------------------------------------
    # Metrics for each threshold method
    # -------------------------------------------------
    threshold_metrics = {}
    for method, thresh_val in [
        ("percentile", thresh_percentile),
        ("roc_optimal", thresh_roc),
        ("risk_based", thresh_risk),
    ]:
        threshold_metrics[method] = compute_metrics_at_threshold(
            recon_error_np, window_labels, thresh_val
        )

    # -------------------------------------------------
    # Standard metrics (AUROC, PR-AUC)
    # -------------------------------------------------
    auroc = compute_auroc(recon_error_np, window_labels)
    pr_auc = compute_pr(recon_error_np, window_labels)

    # -------------------------------------------------
    # Coverage–Risk (error rate) with bootstrapped CIs
    # -------------------------------------------------
    cov, risk_mean, risk_lo, risk_hi = bootstrap_coverage_risk(
        recon_error_np,
        uncertainty_np,
        window_labels,
        steps = 20,
        n_bootstrap = 100,
    )

    # -------------------------------------------------
    # Precision & Recall vs. Coverage (using uncertainty thresholds)
    # -------------------------------------------------
    # Generate uncertainty thresholds (linspace) – handle deterministic case
    unc_min = uncertainty_np.min()
    unc_max = uncertainty_np.max()
    if unc_max - unc_min < 1e-12:
        # All uncertainties are the same – create a small spread
        unc_thresholds = np.linspace(unc_min - 1, unc_min + 1, 20)
    else:
        unc_thresholds = np.linspace(unc_min, unc_max, 20)

    coverages_pr = []
    precisions = []
    recalls = []
    base_thresh = np.percentile(recon_error_np, 95)   # same base as in risk curve

    for t in unc_thresholds:
        keep = uncertainty_np <= t
        if keep.sum() == 0:
            continue
        preds = recon_error_np[keep] > base_thresh
        true = window_labels[keep]
        coverages_pr.append(keep.mean())
        precisions.append(precision_score(true, preds, zero_division = 0))
        recalls.append(recall_score(true, preds, zero_division = 0))

    coverages_pr = np.array(coverages_pr)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # -------------------------------------------------
    # Save all arrays
    # -------------------------------------------------
    np.save(os.path.join(mode_dir, "recon_error.npy"), recon_error_np)
    np.save(os.path.join(mode_dir, "uncertainty.npy"), uncertainty_np)
    np.save(os.path.join(mode_dir, "labels.npy"), window_labels)

    np.save(os.path.join(mode_dir, "coverage.npy"), cov)
    np.save(os.path.join(mode_dir, "risk_mean.npy"), risk_mean)
    np.save(os.path.join(mode_dir, "risk_lo.npy"), risk_lo)
    np.save(os.path.join(mode_dir, "risk_hi.npy"), risk_hi)

    np.save(os.path.join(mode_dir, "coverage_pr.npy"), coverages_pr)
    np.save(os.path.join(mode_dir, "precision_vs_coverage.npy"), precisions)
    np.save(os.path.join(mode_dir, "recall_vs_coverage.npy"), recalls)

    # -------------------------------------------------
    # Save metrics (including threshold_metrics)
    # -------------------------------------------------
    metrics = {
        "seed": seed,
        "mode": mode,
        "AUROC": float(auroc),
        "PR_AUC": float(pr_auc),
        "thresholds": {
            "percentile": float(thresh_percentile),
            "roc_optimal": float(thresh_roc),
            "risk_based": float(thresh_risk),
        },
        "threshold_metrics": threshold_metrics,
    }

    with open(os.path.join(mode_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
    print(f"\nEvaluation complete ({mode}, seed={seed})")
    print(f"AUROC:  {auroc:.4f}")
    print(f"PR_AUC: {pr_auc:.4f}")
    for method, vals in threshold_metrics.items():
        far = vals['false_alarm_rate']
        delay = vals['detection_delay_mean']
        delay_str = f"{delay:.2f}" if delay is not None else "None"
        print(f"{method:12s} -> P={vals['precision']:.3f} R={vals['recall']:.3f} "
              f"F1={vals['f1']:.3f} FAR={far:.3f} Delay={delay_str}")

    return recon_error_np, uncertainty_np, window_labels