import argparse
import torch

from src.utils import set_seed
from src.data_loader import build_datasets
from src.model import LSTMAutoencoder
from src.train import train_autoencoder
from src.evaluate import evaluate
from src.aggregate_results import run_aggregation
from src.visualization import run_visualization


# =================================================
# Single Seed Pipeline
# =================================================
def run_single_seed(seed, device):
    print("\n" + "=" * 60)
    print(f"Running Seed {seed}")
    print("=" * 60)

    set_seed(seed)

    # ---------------------
    # Load data
    # ---------------------
    train_dataset, test_dataset, labels, scaler = build_datasets(
        data_dir="data",
        labels_csv_path="data/labeled_anomalies.csv",
        channel_id="P-1",
        window_size=50,
        stride=1,
    )
    # ---------------------
    # Initialize model
    # ---------------------
    model = LSTMAutoencoder(
        input_dim=train_dataset[0].shape[-1],
        hidden_dim=64,
        latent_dim=16,
        dropout=0.2,
    )
    # ---------------------
    # Train
    # ---------------------
    train_autoencoder(
        model=model,
        train_dataset=train_dataset,
        seed=seed,
        device=device,
    )

    # ---------------------
    # Evaluate MC Dropout
    # ---------------------
    evaluate(
        model=model,
        test_dataset=test_dataset,
        labels=labels,
        seed=seed,
        use_mc_dropout=True,
        device=device,
    )

    # ---------------------
    # Evaluate No Dropout
    # ---------------------
    evaluate(
        model=model,
        test_dataset=test_dataset,
        labels=labels,
        seed=seed,
        use_mc_dropout=False,
        device=device,
    )


# =================================================
# Full Multi-Seed Experiment
# =================================================
def run_full_experiment(num_seeds, device):
    print(f"\nRunning FULL experiment with {num_seeds} seeds...\n")

    for seed in range(num_seeds):
        run_single_seed(seed, device)

    print("\nAggregating results...\n")
    run_aggregation()

    print("\n✓ FULL EXPERIMENT COMPLETE\n")


# =================================================
# CLI Entry
# =================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "aggregate", "plot", "full_experiment"])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    train_dataset, test_dataset, labels, scaler = build_datasets(
        data_dir="data",
        labels_csv_path="data/labeled_anomalies.csv",
        channel_id="P-1",
        window_size=50,
        stride=1,
    )

    model = LSTMAutoencoder(
        input_dim=train_dataset[0].shape[-1],
        hidden_dim=64,
        latent_dim=16,
        dropout=0.2,
    )
    
    # ---------------------
    # Train
    # ---------------------
    if args.mode == "train":
        # run_single_seed(args.seed, device)

        train_autoencoder(
            model=model,
            train_dataset=train_dataset,
            seed=args.seed,
            device=device,
        )

    elif args.mode == "evaluate":
        # evaluate only (assumes checkpoint exists)
        set_seed(args.seed)
        # ---------------------
        # Evaluate MC Dropout
        # ---------------------
        evaluate(
            model=model,
            test_dataset=test_dataset,
            labels=labels,
            seed=args.seed,
            use_mc_dropout=True,
            device=device,
        )

        # ---------------------
        # Evaluate No Dropout
        # ---------------------
        evaluate(
            model=model,
            test_dataset=test_dataset,
            labels=labels,
            seed=args.seed,
            use_mc_dropout=False,
            device=device,
        )

        # print("Generating plots...") 
        # run_visualization(seed=args.seed,)

    elif args.mode == "aggregate":
        run_aggregation()

    elif args.mode == "full_experiment":
        run_full_experiment(args.seeds, device)

    elif args.mode == "plot":
        run_visualization(seed=args.seed)


if __name__ == "__main__":
    main()
