# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import json
import os
from hw2_q2 import utils_w_masking
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

class CNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, use_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.relu = nn.ReLU()
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_classes, use_softmax, conv_params, fc_params, input_size, use_pool):
        super(CNN, self).__init__()

        # Convolutional layers
        in_ch = input_size[0]
        self.convs = nn.ModuleList()
        for out_ch in conv_params:
            self.convs.append(CNNLayer(in_ch, out_ch, kernel=3, stride=1, padding=1, use_pool=use_pool))
            in_ch = out_ch

        H, W = input_size[1], input_size[2]
        if use_pool:
            for _ in conv_params:
                H //= 2  # MaxPool 2
                W //= 2  # MaxPool 2

        # The number of inputs for the first FC layer
        flatten_size = conv_params[-1] * H * W

        # Fully connected layers
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
        labels = labels.squeeze().long().to(device) # ??

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

def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def config_name(use_softmax, use_pool):
    return f"{'softmax' if use_softmax else 'logits'}_{'pool' if use_pool else 'no_pool'}"

def run_experiment(opt, use_softmax, use_pool, train_loader, val_loader, test_loader, n_classes):
    """
    Runs one CNN experiment with a specific configuration.

    Returns:
        train_losses (list)
        val_accs (list)
    """
    # --------- Before Training ----------
    total_start = time.time()

    cfg = config_name(use_softmax, use_pool)

    print(f"\n=== Running experiment: {cfg} ===")

    # Paths
    results_dir = "Q1-CNN-results"
    os.makedirs(results_dir, exist_ok=True) # directory to save results to
    model_path  = os.path.join(results_dir, f"cnn_{cfg}.pth")
    scores_path = os.path.join(results_dir, f"scores_{cfg}.json")

    # Initialize the model
    model = CNN(
        n_classes=n_classes,
        use_softmax=use_softmax,
        conv_params=[32, 64, 128],
        fc_params=[256],
        input_size=(3, 28, 28),
        use_pool=use_pool
    ).to(device)

    # Get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=0
    )

    # Get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_losses, val_accs = [], []
    best_valid = 0.0
    best_epoch = -1
    for ii in epochs:
        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(
            f"[{cfg}] Epoch {ii}/{opt.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f} sec"
        )

        # save the best model checkpoint
        if val_acc > best_valid:
            best_valid = val_acc
            best_epoch = ii
            torch.save(model.state_dict(), model_path)

    # --------- After Training ----------
    total_end = time.time()
    total_time = total_end - total_start

    print(f"\n[{cfg}] Total training time: {total_time/60:.2f} minutes "
        f"({total_time:.2f} seconds)")
    print('Final Test acc: %.4f' % (evaluate(test_loader, model)))

    # Reload best model and evaluate on test set
    best_model = CNN(
        n_classes=n_classes,
        use_softmax=use_softmax,
        conv_params=[32, 64, 128],
        fc_params=[256],
        input_size=(3, 28, 28),
        use_pool=use_pool
    ).to(device)

    best_model.load_state_dict(torch.load(model_path))
    test_acc = evaluate(test_loader, best_model)

    print(
        f"\n[{cfg}] Best epoch: {best_epoch} | "
        f"Best val acc: {best_valid:.4f} | "
        f"Test acc: {test_acc:.4f}"
    )
    print(f"Model saved to {model_path}")

    # Save scores JSON
    scores = {
        "config": {
            "softmax": use_softmax,
            "maxpool": use_pool,
            "learning_rate": opt.learning_rate,
            "optimizer": opt.optimizer
        },
        "best_valid": float(best_valid),
        "selected_epoch": int(best_epoch),
        "test": float(test_acc),
        "time": total_time
    }

    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)

    return train_losses, val_accs


device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Loading
data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

def main(opt):
    train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
    val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
    test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    experiments = [
        {"use_softmax": False, "use_pool": False},
        {"use_softmax": True,  "use_pool": False},
        {"use_softmax": False, "use_pool": True},
        {"use_softmax": True,  "use_pool": True},
    ]

    all_train_losses = {}
    all_val_accs = {}

    for exp in experiments:
        label = config_name(exp["use_softmax"], exp["use_pool"])

        train_losses, val_accs = run_experiment(
            opt,
            exp["use_softmax"],
            exp["use_pool"],
            train_loader,
            val_loader,
            test_loader,
            n_classes
        )

        all_train_losses[label] = train_losses
        all_val_accs[label] = val_accs

    epochs = np.arange(1, opt.epochs + 1)

    plot(
        epochs,
        all_train_losses,
        filename="Q1-CNN-results/CNN-training-loss-all.pdf"
    )

    plot(
        epochs,
        all_val_accs,
        filename="Q1-CNN-results/CNN-validation-accuracy-all.pdf",
        ylim=(0, 1)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=200, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for updates.""")
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils_w_masking.configure_seed(seed=42)

    main(opt)