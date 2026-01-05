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


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


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

# --------- Before Training ----------
total_start = time.time()

def main(opt):
    train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
    val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
    test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    dir_name = "Q1-CNN-results"
    os.makedirs(dir_name, exist_ok=True) # directory to save results to
    model_path = os.path.join(dir_name, opt.save_path)
    scores_path = os.path.join(dir_name, opt.scores)

    # initialize the model
    model = CNN(
        n_classes=n_classes,
        use_softmax=not opt.no_softmax,
        conv_params=[32, 64, 128],
        fc_params=[256],
        input_size=(3, 28, 28),
        use_pool=not opt.no_maxpool
    ).to(device)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=0
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    ### you can use the code below or implement your own loop ###
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

        print(f"Epoch {ii}/{opt.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f} sec")
        
        # save the best model checkpoint
        if val_acc > best_valid:
            best_valid = val_acc
            best_epoch = ii
            torch.save(model.state_dict(), model_path)

    # --------- After Training ----------
    total_end = time.time()
    total_time = total_end - total_start

    print(f"\nTotal training time: {total_time/60:.2f} minutes "
        f"({total_time:.2f} seconds)")
    print('Final Test acc: %.4f' % (evaluate(test_loader, model)))

    print("\nReloading best checkpoint")
    best_model = CNN(
        n_classes=n_classes,
        use_softmax=not opt.no_softmax,
        conv_params=[32, 64, 128],
        fc_params=[256],
        input_size=(3, 28, 28),
        use_pool=not opt.no_maxpool
    ).to(device)
    best_model.load_state_dict(torch.load(model_path))
    test_acc = evaluate(test_loader, best_model)
    print('Best model test acc: {:.4f}'.format(test_acc))
    print(f"Model saved to {model_path}")

    config = f"{opt.learning_rate}-{opt.optimizer}-{opt.no_maxpool}-{opt.no_softmax}"
    config_json = {
        "lr": opt.learning_rate,
        "optimizer": opt.optimizer,
        "maxpool": not opt.no_maxpool,
        "softmax": not opt.no_softmax
    }

    plot(epochs, train_losses, ylabel='Loss', name='Q1-CNN-results/CNN-training-loss-{}'.format(config))
    plot(epochs, val_accs, ylabel='Accuracy', name='Q1-CNN-results/CNN-validation-accuracy-{}'.format(config))

    with open(scores_path, "w") as f:
        json.dump(
            {"config": config_json,
             "best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": total_time},
            f, indent=4
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
    parser.add_argument('-save-path', default='bloodmnist_cnn.pth')
    parser.add_argument('-scores', default="CNN-softmax-scores.json")
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_softmax', action='store_true')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils_w_masking.configure_seed(seed=42)

    main(opt)