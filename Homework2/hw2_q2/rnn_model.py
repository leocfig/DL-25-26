import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import RNAConfig
from utils import load_best_params, masked_mse_loss, masked_spearman_correlation, configure_seed, load_rnacompete_data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional, dropout):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len, alphabet_size)
        rnn_out, _ = self.rnn(x)

        # Mean pooling over sequence length
        pooled = rnn_out.mean(dim=1)
        
        pooled = self.dropout(pooled)

        return self.fc(pooled)

def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_spearmans = []

    model.to(device)

    for epoch in range(num_epochs):
        # -------- Training --------
        model.train()
        train_loss_epoch = 0.0
        n_train_batches = 0

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            preds = model(x)
            loss = masked_mse_loss(preds, y, mask)

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            n_train_batches += 1

        train_loss_epoch /= n_train_batches
        train_losses.append(train_loss_epoch)

        # -------- Validation --------
        model.eval()
        val_loss_epoch = 0.0
        val_spearman_epoch = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)

                preds = model(x)

                loss = masked_mse_loss(preds, y, mask)
                spearman = masked_spearman_correlation(preds, y, mask)

                val_loss_epoch += loss.item()
                val_spearman_epoch += spearman.item()
                n_val_batches += 1

        val_loss_epoch /= n_val_batches
        val_spearman_epoch /= n_val_batches

        val_losses.append(val_loss_epoch)
        val_spearmans.append(val_spearman_epoch)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train MSE: {train_loss_epoch:.4f} | "
            f"Val MSE: {val_loss_epoch:.4f} | "
            f"Val Spearman: {val_spearman_epoch:.4f}"
        )

    return train_losses, val_losses, val_spearmans

def main():
    protein_name = "RBFOX1"
    num_epochs = 30
    alphabet_size = 4

    # ---------------- Load best hyperparameters ----------------
    best_params = load_best_params("best_rnn_params.json")

    hidden_size = best_params["hidden_size"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["lr"]
    dropout = best_params["dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Setting seed for reproducibility ---------
    configure_seed(RNAConfig.SEED)

    train_dataset = load_rnacompete_data(protein_name, split="train")
    val_dataset = load_rnacompete_data(protein_name, split="val")
    test_dataset = load_rnacompete_data(protein_name, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RNN(
        input_size=alphabet_size,
        hidden_size=hidden_size,
        output_size=1,
        bidirectional=True,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------- Training ----------------
    train_losses, val_losses, val_spearmans = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # ---------------- Saving Model ----------------
    torch.save(model.state_dict(), "rnn_rbfox1.pt")

    # ---------------- Testing ----------------
    model.eval()
    test_loss = 0.0
    test_spearman = 0.0
    n_test_batches = 0

    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            preds = model(x)
            loss = masked_mse_loss(preds, y, mask)
            spearman = masked_spearman_correlation(preds, y, mask)

            test_loss += loss.item()
            test_spearman += spearman.item()
            n_test_batches += 1

    test_loss /= n_test_batches
    test_spearman /= n_test_batches

    print(f"Test MSE: {test_loss:.4f} | Test Spearman: {test_spearman:.4f}")

    print("Training and testing finished.")

if __name__ == "__main__":
    main()