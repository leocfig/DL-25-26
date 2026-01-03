import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import RNAConfig
from utils_w_masking import load_best_params, masked_mse_loss, masked_spearman_correlation, configure_seed, load_rnacompete_data, plot, reshape_tensor_dataset

class CNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, use_pool, dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.use_pool:
            x = self.pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, conv_params, fc_params, input_size, kernel_size, use_pool, dropout):
        super(CNN, self).__init__()

        # Convolutional layers
        in_ch = input_size[0]
        self.convs = nn.ModuleList()
        for out_ch in conv_params:
            self.convs.append(CNNLayer(in_ch, out_ch, kernel=kernel_size, stride=1, padding=kernel_size // 2, use_pool=use_pool, dropout=dropout))
            in_ch = out_ch

        # Infer flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            for conv in self.convs:
                dummy = conv(dummy)
            flatten_size = dummy.view(1, -1).size(1)

        fc_sizes = [flatten_size] + fc_params + [1]  # output = 1 (regression)
        self.fcs = nn.ModuleList()
        for i in range(len(fc_sizes)-1):
            self.fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, x_mask):
        # Apply mask to zero out padded positions
        x = x * x_mask.unsqueeze(1)  # (B, 1, seq_len, 4), broadcast over channel

        # Pass through convolutional layers
        for conv in self.convs:
            x = conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs)-1:
                x = self.dropout(x)
                x = self.activation(x)

        return x


def train_epoch(loader, model, optimizer):
    """
    Train one epoch, updating weights with the given batch.
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
    Returns:
        mean loss (float)
    """
    model.train()
    total_loss = 0
    for x, x_mask, y, mask in loader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        mask = mask.to(device)

        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Forward pass
        preds = model(x, x_mask)
        # Compute the loss between predicted outputs and true labels
        loss = masked_mse_loss(preds, y, mask)
        # Backpropagate the loss: compute gradients
        loss.backward()
        # Update model parameters using the computed gradients
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(loader, model):
    model.eval()
    total_spearman = 0
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    protein_name = "RBFOX1"
    num_epochs = 30
    alphabet_size = 4

    # ---------------- Load best hyperparameters ----------------
    best_params = load_best_params("optuna_results/best_cnn_params.json")

    kernel_size = best_params["kernel_size"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["lr"]
    dropout = best_params["dropout"]
    conv_params = best_params["conv_params"]
    fc_params = best_params["fc_params"]
    no_maxpool = best_params["no_maxpool"]

    # ---------------- Setting seed for reproducibility ---------
    configure_seed(RNAConfig.SEED)

    # ---------------- Loading Data ----------------
    train_dataset = load_rnacompete_data(protein_name, split="train")
    val_dataset = load_rnacompete_data(protein_name, split="val")
    test_dataset = load_rnacompete_data(protein_name, split="test")

    train_dataset = reshape_tensor_dataset(train_dataset)
    val_dataset = reshape_tensor_dataset(val_dataset)
    test_dataset = reshape_tensor_dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(
        conv_params=conv_params,
        fc_params=fc_params,
        input_size=(1, RNAConfig.SEQ_MAX_LEN, alphabet_size),
        kernel_size=kernel_size,
        use_pool=not no_maxpool,
        dropout=dropout
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------- Training ----------------
    train_losses = []
    val_spearman = []
    epochs = np.arange(1, num_epochs + 1)
    for ii in epochs:
        train_loss = train_epoch(train_loader, model, optimizer)
        val_spearman_value = evaluate(val_loader, model)
        train_losses.append(train_loss)
        val_spearman.append(val_spearman_value)
        print(f"Epoch {ii}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Spearman: {val_spearman_value:.4f}")

    # ---------------- Saving Model ----------------
    torch.save(model.state_dict(), "cnn_rbfox1.pt")
    print("Model saved as cnn_rbfox1.pt")

    # ---------------- Testing --------------------
    test_spearman = evaluate(test_loader, model)
    print(f"Test Spearman: {test_spearman:.4f}")

    # ---------------- Plotting --------------------
    conv_str = "-".join(map(str, conv_params))
    fc_str = "-".join(map(str, fc_params))
    config = f"{batch_size}-{learning_rate}-{conv_str}-{fc_str}"

    os.makedirs("Q2-CNN-results", exist_ok=True) # directory to save results to

    plottables = {
        "Train Loss": train_losses,
        "Val Spearman": val_spearman
    }
    
    plot(epochs, plottables, filename=f'Q2-CNN-results/CNN-training-plot-{config}.pdf')

    print("Training and testing finished.")

if __name__ == "__main__":
    main()