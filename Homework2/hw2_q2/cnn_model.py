import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import masked_mse_loss, masked_spearman_correlation, configure_seed, load_rnacompete_data

class CNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):
    def __init__(self, conv_params, fc_params, input_size):
        super(CNN, self).__init__()

        # Convolutional layers
        in_ch = input_size[0]
        self.convs = nn.ModuleList()
        for out_ch in conv_params:
            self.convs.append(CNNLayer(in_ch, out_ch, kernel=3, stride=1, padding=1))
            in_ch = out_ch

        # Fully connected layers
        H, W = input_size[1], input_size[2]
        flatten_size = conv_params[-1] * H * W 
        fc_sizes = [flatten_size] + fc_params + [1]  # output = 1 (regression)
        self.fcs = nn.ModuleList()
        for i in range(len(fc_sizes)-1):
            self.fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))

        self.activation = nn.ReLU()

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs)-1:
                x = self.activation(x)

        return x


def train_epoch(loader, model, optimizer):
    """
    Train one epoch, updating weights with the given batch.
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        criterion: loss function
        optimizer: optimizer used in gradient step
    Returns:
        mean loss (float)
    """
    model.train()
    total_loss = 0
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Forward pass
        preds = model(x)
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
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            preds = model(x)
            spearman = masked_spearman_correlation(preds, y, mask)
            total_spearman += spearman.item()
            n_batches += 1

    return total_spearman / n_batches

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # ---------------- CONFIG ----------------
    # MUDAR PARAMETROS AQUI
    protein_name = "RBFOX1"
    batch_size = 64
    num_epochs = 30
    learning_rate = 1e-3
    seed = 42

    conv_params = [16, 32]
    fc_params = [64]
    seq_len = 41
    alphabet_size = 4

    configure_seed(seed)

    # ---------------- Load Data ----------------
    train_dataset = load_rnacompete_data(protein_name, split="train")
    val_dataset = load_rnacompete_data(protein_name, split="val")
    test_dataset = load_rnacompete_data(protein_name, split="test")

    def reshape_tensor_dataset(ds):
        x_list, y_list, mask_list = [], [], []
        for x, y, mask in ds:
            x_list.append(x.unsqueeze(0))  # (1, seq_len, 4)
            y_list.append(y)
            mask_list.append(mask)
        # Concatena tudo
        x_tensor = torch.stack(x_list)
        y_tensor = torch.stack(y_list)
        mask_tensor = torch.stack(mask_list)
        return TensorDataset(x_tensor, y_tensor, mask_tensor)

    train_dataset = reshape_tensor_dataset(train_dataset)
    val_dataset = reshape_tensor_dataset(val_dataset)
    test_dataset = reshape_tensor_dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(
        conv_params=conv_params,
        fc_params=fc_params,
        input_size=(1, seq_len, alphabet_size)
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------- TRAIN ----------------
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, model, optimizer)
        val_spearman = evaluate(val_loader, model)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Spearman: {val_spearman:.4f}")

    # ---------------- SAVE MODEL ----------------
    torch.save(model.state_dict(), "rnn_rbfox1.pt")

    # ---------------- TEST ----------------
    test_spearman = evaluate(test_loader, model)
    print(f"Test Spearman: {test_spearman:.4f}")

    print("Training and testing finished.")

if __name__ == "__main__":
    main()