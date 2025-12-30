import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import masked_mse_loss, masked_spearman_correlation, configure_seed, load_rnacompete_data

class RNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # Input + Hidden → Hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        # Hidden → Output (regressão)
        self.h2o = nn.Linear(hidden_size, output_size)

        # Hidden state
        self.initial_hidden = nn.Parameter(
            torch.zeros(hidden_size)
        )

    def single_step(self, x_t, hidden):
        """
        x_t: (B, input_size)
        hidden: (B, hidden_size)
        """
        combined = torch.cat((x_t, hidden), dim=1)
        hidden = self.tanh(self.i2h(combined))
        return hidden

    def forward(self, x):
        """
        x: (B, seq_len, 4)
        returns: (B, 1)
        """

        batch_size, seq_len, _ = x.size()

        hidden = self.initial_hidden.unsqueeze(0).expand(
            batch_size, self.hidden_size
        )

        hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]               # (B, 4)
            hidden = self.single_step(x_t, hidden)
            hidden_states.append(hidden)

        # Stack → (B, seq_len, hidden_size)
        hidden_states = torch.stack(hidden_states, dim=1)

        # Mean pooling 
        pooled = hidden_states.mean(dim=1)  # (B, hidden_size)

        # Regression
        output = self.h2o(pooled)            # (B, 1)

        return output

def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_spearmans = []

    model.to(device)

    for epoch in range(num_epochs):
        # -------- TRAIN --------
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

        # -------- VALIDATION --------
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
    # ---------------- CONFIG ----------------
    # MUDAR PARAMETROS AQUI
    protein_name = "RBFOX1"
    batch_size = 64
    num_epochs = 30
    learning_rate = 1e-3
    hidden_size = 128
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure_seed(seed)

    train_dataset = load_rnacompete_data(protein_name, split="train")
    val_dataset = load_rnacompete_data(protein_name, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RNN(
        input_size=4,
        hidden_size=hidden_size,
        output_size=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------- TRAIN ----------------
    train_losses, val_losses, val_spearmans = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # ---------------- SAVE MODEL ----------------
    torch.save(model.state_dict(), "rnn_rbfox1.pt")

    # ---------------- TEST ----------------
    test_dataset = load_rnacompete_data(protein_name, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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