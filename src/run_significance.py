import os
import json
import numpy as np

from statistics import significance_test

RESULTS_DIR = "results"


def run_significance():
    """
    Run statistical significance testing between
    MC Dropout and No Dropout experiments.
    """

    mc = np.load(os.path.join(RESULTS_DIR, "mc_dropout", "recon_error.npy"))
    no = np.load(os.path.join(RESULTS_DIR, "no_dropout", "recon_error.npy"))

    results = significance_test(mc, no)

    with open(os.path.join(RESULTS_DIR, "significance.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Statistical significance results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    run_significance()
