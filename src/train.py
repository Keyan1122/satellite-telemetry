import os

import torch
from torch.utils.data import DataLoader
from .model import reconstruction_loss

def train_autoencoder(
    model,
    train_dataset,
    seed,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device="cpu",
):
    model.to(device)
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining Seed {seed}...\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for x in loader:
            x = x.to(device)

            optimizer.zero_grad()
            x_recon = model(x)
            loss = reconstruction_loss(x, x_recon)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.6f}")

    # ---------------------
    # Save checkpoint per seed
    # ---------------------
    os.makedirs("checkpoints", exist_ok=True)

    save_path = f"checkpoints/model_seed_{seed}.pt"
    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to {save_path}\n")