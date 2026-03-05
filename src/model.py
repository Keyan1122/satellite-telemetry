import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers = 1, dropout = 0.0):
        """
        LSTM Autoencoder for multivariate time-series.

        Args:
            input_dim (int): Number of features per timestep
            hidden_dim (int): LSTM hidden size
            latent_dim (int): Bottleneck dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability (used later for MC Dropout)
        """
        super().__init__()

        # -------- Encoder --------
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout if num_layers > 1 else 0.0)
        self.encoder_dropout = nn.Dropout(dropout)

        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # -------- Decoder --------
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers = num_layers, batch_first = True, dropout = dropout if num_layers > 1 else 0.0)
        self.decoder_dropout = nn.Dropout(dropout)


        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            x_recon: reconstructed input
        """
        batch_size, seq_len, _ = x.size()

        # ---- Encoder ----
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # (batch, hidden_dim)
        h_last = self.encoder_dropout(h_last)   # MC Dropout point

        z = self.encoder_fc(h_last)  # latent vector

        # ---- Decoder ----
        hidden = self.decoder_fc(z)

        seq_len = x.size(1)
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder_lstm(decoder_input)
        dec_out = self.decoder_dropout(dec_out)
        x_recon = self.output_layer(dec_out)

        return x_recon

def reconstruction_loss(x, x_recon):
    return torch.mean((x - x_recon) ** 2)
