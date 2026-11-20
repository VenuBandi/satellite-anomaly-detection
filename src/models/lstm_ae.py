"""LSTM Autoencoder for satellite telemetry anomaly detection."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers,
                                batch_first=True, dropout=0.2)

    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        # repeat last hidden state across sequence length
        dec_input = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        return dec_out


def create_sequences(data: np.ndarray, seq_len: int = 64) -> np.ndarray:
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i + seq_len])
    return np.array(seqs, dtype=np.float32)


def train_autoencoder(
    train_data: np.ndarray,
    seq_len: int = 64,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple["LSTMAutoencoder", list[float]]:
    seqs = create_sequences(train_data, seq_len)
    if seqs.ndim == 2:
        seqs = seqs[:, :, np.newaxis]
    input_dim = seqs.shape[2]

    dataset = TensorDataset(torch.tensor(seqs))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
    return model, losses


def reconstruction_errors(
    model: "LSTMAutoencoder",
    data: np.ndarray,
    seq_len: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    seqs = create_sequences(data, seq_len)
    if seqs.ndim == 2:
        seqs = seqs[:, :, np.newaxis]
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(len(seqs)):
            x = torch.tensor(seqs[i:i+1]).to(device)
            recon = model(x)
            err = ((x - recon) ** 2).mean().item()
            errors.append(err)
    return np.array(errors)
