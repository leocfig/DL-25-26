from sklearn.base import accuracy_score
import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_classes, use_softmax, conv_params, fc_params, input_size):
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
        fc_sizes = [flatten_size] + fc_params + [n_classes]
        self.fcs = nn.ModuleList()
        for i in range(len(fc_sizes)-1):
            self.fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))

        self.activation = nn.ReLU()

        # Optional Softmax
        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax = nn.Softmax(dim=1)

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

        # Softmax opcional
        if self.use_softmax:
            x = self.softmax(x)

        return x


def train_epoch(loader, model, criterion, optimizer):
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
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Forward pass
        logits = model(imgs)
        # Compute the loss between predicted outputs and true labels
        loss = criterion(logits, labels)
        # Backpropagate the loss: compute gradients
        loss.backward()
        # Update model parameters using the computed gradients
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)

device = "cuda" if torch.cuda.is_available() else "cpu"
