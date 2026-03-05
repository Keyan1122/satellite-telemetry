import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import roc_curve

# -------------------------------------------------
# Threshold selection methods
# -------------------------------------------------

def percentile_threshold(scores, percentile = 95):
    """
    Simple unsupervised threshold.
    """
    scores = np.asarray(scores)
    return np.percentile(scores, percentile)


def roc_optimal_threshold(scores, labels):
    """
    Supervised threshold using ROC curve (Youden's J statistic).
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    return thresholds[best_idx]


def risk_based_threshold(scores, uncertainty, alpha = 0.95):
    """
    Risk-aware threshold: penalize uncertain predictions.
    """
    scores = np.asarray(scores)
    uncertainty = np.asarray(uncertainty)

    adjusted_score = scores + alpha * uncertainty
    return np.percentile(adjusted_score, 95)


# -------------------------------------------------
# Quantitative metrics
# -------------------------------------------------

def compute_auroc(scores, labels):
    """
    Area Under ROC Curve.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    return roc_auc_score(labels, scores)


def compute_pr(scores, labels):
    """
    Precision–Recall AUC.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


# -------------------------------------------------
# False alarm rate
# -------------------------------------------------
def false_alarm_rate(labels, preds):
    """
    Compute false alarm rate = FP / (FP + TN).
    """
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    if tn + fp == 0:
        return 0.0
    return fp / (tn + fp)


# -------------------------------------------------
# Detection delay (in windows) for each anomaly event
# -------------------------------------------------
def detection_delay(labels, preds):
    """
    Compute mean detection delay over all anomaly events.

    Args:
        labels: ground truth binary labels (1 = anomaly, 0 = normal)
        preds:  predicted binary labels (1 = anomaly, 0 = normal)

    Returns:
        mean_delay: average delay (in windows) for detected events
        missed_events: number of anomaly events not detected
        total_events: total number of anomaly events
    """
    # Find contiguous anomaly events in labels
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    delays = []
    missed = 0
    total_events = len(starts)

    for start, end in zip(starts, ends):
        # Find first positive prediction within [start, end]
        event_preds = preds[start:end+1]
        pos_indices = np.where(event_preds == 1)[0]
        if len(pos_indices) > 0:
            first_detection = start + pos_indices[0]
            delay = first_detection - start
            delays.append(delay)
        else:
            missed += 1

    mean_delay = np.mean(delays) if delays else np.nan
    return mean_delay, missed, total_events


# -------------------------------------------------
# Coverage–Risk analysis
# -------------------------------------------------

def coverage_risk_curve(scores, uncertainty, labels, steps = 20):
    """
    Compute coverage–risk trade-off.

    Coverage = fraction of samples kept
    Risk = error rate on kept samples
    """
    scores = np.asarray(scores)
    uncertainty = np.asarray(uncertainty)
    labels = np.asarray(labels)

    coverages = []
    risks = []

    thresholds = np.linspace(
        uncertainty.min(),
        uncertainty.max(),
        steps
    )
    
    base_threshold = np.percentile(scores, 95)

    for t in thresholds:
        keep = uncertainty <= t

        if keep.sum() == 0:
            continue

        preds = scores[keep] > base_threshold
        true = labels[keep]

        risk = np.mean(preds != true)
        coverage = keep.mean()

        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)

def bootstrap_coverage_risk(
    scores,
    uncertainty,
    labels,
    steps=20,
    n_bootstrap=100,
    alpha=0.95,
):
    """
    Bootstrap confidence intervals for coverage–risk curve.
    Returns mean risk and confidence bounds at each coverage.
    """

    rng = np.random.default_rng(seed = 42)
    risks_all = []

    N = len(scores)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)

        cov, risk = coverage_risk_curve(
            scores[idx],
            uncertainty[idx],
            labels[idx],
            steps=steps,
        )

        risks_all.append(risk)

    risks_all = np.array(risks_all)

    mean_risk = risks_all.mean(axis=0)
    lower = np.percentile(risks_all, (1 - alpha) / 2 * 100, axis=0)
    upper = np.percentile(risks_all, (1 + alpha) / 2 * 100, axis=0)

    return cov, mean_risk, lower, upper
