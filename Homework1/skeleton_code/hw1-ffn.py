#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils
import json
import os

class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()
        
        activations = {"tanh": nn.Tanh(), "relu": nn.ReLU()}
        activation = activations[activation_type]

        dropout = nn.Dropout(dropout)

        layer_input_dims = [n_features] + [hidden_size] * layers
        layer_output_dims = [hidden_size] * layers + [n_classes]

        hidden_layers = []
        for input_dim, output_dim in zip(layer_input_dims[:-1], layer_output_dims[:-1]):
            block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                activation,
                dropout
            )
            hidden_layers.append(block)

        all_layers = hidden_layers + [nn.Linear(layer_input_dims[-1], layer_output_dims[-1])]

        self.feedforward = nn.Sequential(*all_layers)

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        return self.feedforward(x)
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    optimizer.zero_grad()
    logits = model(X, **kwargs)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    logits = model(X)
    preds = logits.argmax(dim=-1)
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)
    loss = loss.item()
    preds = logits.argmax(dim=-1)
    accuracy = (y == preds).float().mean().item()
    model.train()
    return loss, accuracy

def grid_search(n_classes, n_feats, widths, learning_rates, dropouts, weight_decays,
                train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
                epochs=30, activation="relu", optimizer="sgd"):
    results = []

    best_model_overall = None
    best_val_acc_overall = 0
    best_history_overall = None

    criterion = nn.CrossEntropyLoss()

    print("Performing grid search...\n")
    os.makedirs("ffn_grid_search_results", exist_ok=True) # directory to save results to

    train_acc_width = []

    for width in widths:
        best_config_width = None
        best_val_acc_width = 0
        best_train_acc_for_width = 0
        for lr in learning_rates:
            for dp in dropouts:
                for wd in weight_decays:
                    print(f"Running config: width={width}, lr={lr}, dropout={dp}, wd={wd}")
                    model = FeedforwardNetwork(n_classes, n_feats, width, 1, activation, dp)
                    optimizer_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
                    optimizer_inst = optimizer_cls(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wd
                    )
                    # Saving metrics per configuration
                    best_val_acc_conf = 0
                    train_losses_conf = []
                    train_accs_conf = []
                    valid_losses_conf = []
                    valid_accs_conf = []

                    config_start_time = time.time()
                    for epoch in range(epochs):
                        epoch_train_losses = []
                        model.train()
                        for X_batch, y_batch in train_dataloader:
                            loss = train_batch(X_batch, y_batch, model, optimizer_inst, criterion)
                            epoch_train_losses.append(loss)

                        model.eval()
                        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
                        _, train_acc = evaluate(model, train_X, train_y, criterion)
                        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)
                        best_val_acc_conf = max(best_val_acc_conf, val_acc)

                        train_losses_conf.append(epoch_train_loss)
                        train_accs_conf.append(train_acc)
                        valid_losses_conf.append(val_loss)
                        valid_accs_conf.append(val_acc)

                    config_runtime = time.time() - config_start_time
                    config = { 
                        "width": width,
                        "lr": lr,
                        "dropout": dp,
                        "weight_decay": wd,
                        "val_acc": best_val_acc_conf,
                        "runtime": config_runtime
                    }

                    if best_val_acc_conf > best_val_acc_overall:
                        best_val_acc_overall = best_val_acc_conf
                        best_model_overall = model
                        best_history_overall = {
                            "train_losses": train_losses_conf.copy(),
                            "train_accs": train_accs_conf.copy(),
                            "valid_losses": valid_losses_conf.copy(),
                            "valid_accs": valid_accs_conf.copy(),
                            "config": config
                        }
                    
                    if best_val_acc_width < best_val_acc_conf:
                        best_val_acc_width = best_val_acc_conf
                        best_config_width = config
                        train_acc_for_best_val = train_accs_conf[valid_accs_conf.index(best_val_acc_conf)]
                        best_train_acc_for_width = train_acc_for_best_val

                    results.append(config)
                    print(f" → best val acc = {best_val_acc_conf:.4f}")

        train_acc_width.append(best_train_acc_for_width)
        print(f"\n→ Best config for width {width}: {best_config_width}\n")

        # Save JSON per width with the best model configuration
        width_save_path = f"ffn_grid_search_results/best_config_width_{width}.json"
        with open(width_save_path, "w") as f:
            json.dump(best_config_width, f, indent=4)

    print("\n=================== GRID SEARCH RESULTS ===================")
    for r in results:
        print(r)
    print("============================================================\n")
    # save results to a file
    output_path = "ffn_grid_search_results/grid_search_results.txt"
    with open(output_path, "w") as f:
        for r in results:
            f.write(str(r) + "\n")

    # Evaluate on test set the best model configuration
    _, test_acc = evaluate(best_model_overall, test_X, test_y, criterion)
    print(f"Test accuracy of best model: {test_acc:.4f}")

    # Plot training loss and validation accuracy
    epochs_range = list(range(1, epochs+1))
    plot(epochs_range, {
        "Train Loss": best_history_overall["train_losses"]
    }, filename="ffn_grid_search_results/ffn_best_train_loss.pdf")

    plot(epochs_range, {
        "Valid Accuracy": best_history_overall["valid_accs"]
    }, filename="ffn_grid_search_results/ffn_best_val_acc.pdf")

    utils.plot(
        "Hidden Layer Width",
        "Training Accuracy",
        {"Train Accuracy": (widths, train_acc_width)},
        filename="ffn_grid_search_results/ffn_train_acc_vs_width.pdf"
    )

def depth_experiment(n_classes, n_feats, train_dataloader, train_X, train_y, dev_X, dev_y, test_X, test_y,
                     epochs, depths, hidden_size, best_config, activation="relu", optimizer="sgd"):
    results = {}

    # Extract best hyperparameters from the stored config
    lr = best_config["lr"]
    dropout = best_config["dropout"]
    weight_decay = best_config["weight_decay"]
    criterion = nn.CrossEntropyLoss()

    best_model_overall = None
    best_val_acc_overall = 0
    best_history_overall = None
    final_train_accs = []

    os.makedirs("ffn_depth_results", exist_ok=True) # directory to save results to
    print("\n[DEPTH EXPERIMENT] Starting depth sweep...")
    print(f"Using best hyperparameters from width=32: {best_config}\n")

    for L in depths:
        print(f"\n[DEPTH EXPERIMENT] Training model with L={L} layers")

        model = FeedforwardNetwork(
            n_classes=n_classes,
            n_features=n_feats,
            hidden_size=hidden_size,
            layers=L,
            activation_type=activation,
            dropout=dropout
        )

        optimizer_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
        optimizer_inst = optimizer_cls(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        best_val_acc = -1
        train_losses = []
        val_accs = []

        for epoch in range(epochs):
            epoch_train_losses = []
            model.train()
            for X_batch, y_batch in train_dataloader:
                loss = train_batch(X_batch, y_batch, model, optimizer_inst, criterion)
                epoch_train_losses.append(loss)

            model.eval()
            epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
            _, train_acc = evaluate(model, train_X, train_y, criterion)
            val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

            # Save training losses and validation accuracies for plotting
            train_losses.append(epoch_train_loss)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f'[L={L}] Epoch {epoch+1}/{epochs} | train loss: {epoch_train_loss:.4f} | \
train_acc: {train_acc:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}')

        results[L] = best_val_acc
        final_train_accs.append(train_acc)
        if best_val_acc > best_val_acc_overall:
            best_val_acc_overall = best_val_acc
            best_model_overall = model
            best_history_overall = {
                "train_losses": train_losses.copy(),
                "valid_accs": val_accs.copy()
            }

    # Build results table
    lines = []
    lines.append("-------------------------------------")
    lines.append(" Depth | Best Validation Accuracy")
    lines.append("-------------------------------------")
    for L in depths:
        lines.append(f"   {L:<4} | {results[L]:.4f}")
    lines.append("-------------------------------------")
    table_text = "\n".join(lines)

    # Print to console
    print("\n[DEPTH EXPERIMENT] Summary:")
    print(table_text)

    # Save to file
    output_path = "ffn_depth_results/depth_val_results.txt"
    with open(output_path, "w") as f:
        f.write(table_text + "\n")


    # Evaluate on test set the best model configuration
    _, test_acc = evaluate(best_model_overall, test_X, test_y, criterion)
    print(f"Test accuracy of best model: {test_acc:.4f}")

    # Plot training loss and validation accuracy over epochs
    epochs_range = list(range(1, epochs+1))
    plot(epochs_range, {
        "Train Loss": best_history_overall["train_losses"]
    }, filename="ffn_depth_results/ffn_best_train_losses.pdf")

    plot(epochs_range, {
        "Validation Accuracy": best_history_overall["valid_accs"]
    }, filename="ffn_depth_results/ffn_best_val_accs.pdf")

    # Plot training accuracy as a function of depth
    utils.plot(
        "Network Depth",
        "Training Accuracy",
        {"Train Accuracy": (depths, final_train_accs)},
        filename="ffn_depth_results/ffn_train_acc_vs_depth.pdf"
    )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
    # parser.add_argument('-grid_search', action='store_true',
    #                     help="Run hyperparameter grid search instead of single run.")
    parser.add_argument('-mode', choices=['single', 'grid', 'depth'], default='single',
                        help="Choose experiment: 'single' for 2.1, 'grid' for 2.2, 'depth' for 2.3")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")


    if opt.mode == "grid":
        widths = [16, 32, 64, 128, 256]
        # ESCOLHI ESTES VALORES MAS PODEMOS MUDAR
        learning_rates = [0.1, 0.01, 0.005, 0.001]
        dropouts = [0.0, 0.2]
        weight_decays = [0.0, 1e-4]
        grid_search(
            n_classes, n_feats, widths, learning_rates, dropouts, weight_decays,
            train_dataloader, train_X, train_y,
            dev_X, dev_y, test_X, test_y,
            epochs=30,
            activation="relu",
            optimizer="sgd"
        )
        return
    elif opt.mode == "depth":
        depths = [1, 3, 5, 7, 9]
        # Load best config from JSON file
        with open("ffn_grid_search_results/best_config_width_32.json", "r") as f:
            best_config = json.load(f)

        depth_experiment(
            n_classes, n_feats,
            train_dataloader, train_X, train_y,
            dev_X, dev_y, test_X, test_y,
            epochs=30, depths=depths, hidden_size=32,
            best_config=best_config,
            activation="relu",
            optimizer="sgd"
        )
        return

    # initialize the model
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs[1:]:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f}| train_acc: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, train_acc, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    accs = {
        "Train Accuracy": train_accs,
        "Valid Accuracy": valid_accs
    }

    plot(epochs, losses, filename=f'losses-{config}.pdf')
    plot(epochs, accs, filename=f'accs-{config}.pdf')
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accs):.4f}")


if __name__ == '__main__':
    main()
