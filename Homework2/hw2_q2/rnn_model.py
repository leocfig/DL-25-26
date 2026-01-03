import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from config import RNAConfig
from utils_w_masking import load_best_params, masked_mse_loss, masked_spearman_correlation, configure_seed, load_rnacompete_data, plot

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional, dropout, use_attention=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_size * self.num_directions

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention Layers
        if self.use_attention:
            self.attn_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.attn_score = nn.Linear(self.hidden_dim, 1, bias=False) # scoring vector v

        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        # x: (B, seq_len, alphabet_size)
        rnn_out, _ = self.rnn(x)

        if self.use_attention:
            # Additive attention pooling
            # (B, seq_len, hidden_dim)
            attn_hidden = torch.tanh(self.attn_fc(rnn_out))

            # (B, seq_len, 1) -> (B, seq_len)
            attn_scores = self.attn_score(attn_hidden).squeeze(-1)

            # Mask padding positions
            attn_scores = attn_scores.masked_fill(x_mask == 0, -1e9)

            # Normalize over sequence length (convert to probabilities)
            attn_weights = torch.softmax(attn_scores, dim=1)

            # Weighted sum of hidden states
            pooled = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)
        else:
            # x_mask: (B, seq_len) → (B, seq_len, 1)
            mask = x_mask.unsqueeze(-1)

            # Zero out padded positions
            masked_rnn_out = rnn_out * mask

            # Sum only valid positions
            sum_hidden = masked_rnn_out.sum(dim=1)

            # Divide by number of valid tokens
            lengths = mask.sum(dim=1).clamp(min=1)

            # Mean pooling
            pooled = sum_hidden / lengths

        pooled = self.dropout(pooled)
        return self.fc(pooled)

def train_epoch(loader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, x_mask, y, mask in loader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        preds = model(x, x_mask)
        loss = masked_mse_loss(preds, y, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(loader, model, device):
    model.eval()
    total_spearman = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, x_mask, y, mask in loader:
            x = x.to(device)
            x_mask = x_mask.to(device)
            y = y.to(device)
            mask = mask.to(device)

            preds = model(x, x_mask)
            spearman = masked_spearman_correlation(preds, y, mask)

            total_spearman += spearman.item()
            n_batches += 1

    return total_spearman / n_batches

def main(opt):
    protein_name = "RBFOX1"
    num_epochs = 30
    alphabet_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = load_best_params("optuna_results/best_rnn_params.json")

    hidden_size = best_params["hidden_size"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["lr"]
    dropout = best_params["dropout"]
    bidirectional = best_params["bidirectional"]

    # ---------------- Setting seed ----------------
    configure_seed(RNAConfig.SEED)

    # ---------------- Loading Data ----------------
    train_dataset = load_rnacompete_data(protein_name, split="train")
    val_dataset = load_rnacompete_data(protein_name, split="val")
    test_dataset = load_rnacompete_data(protein_name, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---------------- Model ----------------
    model = RNN(
        input_size=alphabet_size,
        hidden_size=hidden_size,
        output_size=1,
        bidirectional=bidirectional,
        dropout=dropout,
        use_attention=opt.use_attention
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------- Training ----------------
    train_losses = []
    val_spearman = []
    epochs = np.arange(1, num_epochs + 1)
    for epoch in epochs:
        train_loss = train_epoch(train_loader, model, optimizer, device)
        val_spearman_value = evaluate(val_loader, model, device)

        train_losses.append(train_loss)
        val_spearman.append(val_spearman_value)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val Spearman: {val_spearman_value:.4f}"
        )

    # ---------------- Saving Model ----------------
    torch.save(model.state_dict(), "rnn_rbfox1.pt")
    print("Model saved as rnn_rbfox1.pt")

    # ---------------- Testing ----------------
    test_spearman = evaluate(test_loader, model, device)
    print(f"Test Spearman: {test_spearman:.4f}")

    # ---------------- Plotting --------------------
    config = f"{batch_size}-{learning_rate}-{hidden_size}-bi{int(bidirectional)}-drop{dropout}"
    os.makedirs("Q2-RNN-results", exist_ok=True)

    plottables = {
        "Train Loss": train_losses,
        "Val Spearman": val_spearman
    }

    plot(
        epochs,
        plottables,
        filename=f"Q2-RNN-results/RNN-training-plot-{config}.pdf"
    )

    print("Training and testing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_attention', action='store_true')

    opt = parser.parse_args()
    main(opt)