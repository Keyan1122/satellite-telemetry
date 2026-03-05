import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def cohens_d(x, y):
    """
    Paired Cohen's d effect size.
    """
    diff = x - y
    return float(diff.mean() / diff.std(ddof=1))


def significance_test(x, y, alpha=0.05):
    """
    Run paired statistical significance tests
    between two experimental conditions.
    """

    # Paired t-test
    t_stat, p_t = ttest_rel(x, y)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, p_w = wilcoxon(x, y)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    d = cohens_d(x, y)

    return {
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_t),
            "significant": bool(p_t < alpha),
        },
        "wilcoxon_test": {
            "w_statistic": float(w_stat),
            "p_value": float(p_w) if not np.isnan(p_w) else None,
            "significant": bool(p_w < alpha) if not np.isnan(p_w) else None,
        },
        "effect_size": {
            "cohens_d": float(d),
            "interpretation": (
                "negligible" if abs(d) < 0.2 else
                "small" if abs(d) < 0.5 else
                "medium" if abs(d) < 0.8 else
                "large"
            ),
        },
    }
