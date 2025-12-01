#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import itertools
import time
import pickle
import json
import numpy as np
import utils


class LogisticRegression:
    def __init__(self, n_classes, n_features, eta=0.0001, regularization=0.00001):
        self.W = np.zeros((n_classes, n_features))
        self.eta = eta
        self.regularization = regularization

    def save(self, path):
        """
        Save logistic regression model to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load logistic regression model from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_weight(self, x_i, y_one_hot, label_probabilities):
        """
        x_i (n_features,): a single training example
        y_one_hot (num_classes x 1): one-hot vector of the true class for that example
        label_probabilities (num_classes x 1): softmax probabilities for that example
        """
        self.W += self.eta * ((y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis=1).T) - self.regularization * self.W)

    def train_epoch(self, inputs, labels):
        """
        inputs (n_examples, n_features + 1): observations array of the dataset accounting for bias
        labels (n_examples,): array of target values of the dataset
        """
        # For each observation in data
        for x, y in zip(inputs, labels):
            
            # Get probability scores according to the model
            label_scores = np.expand_dims(self.W.dot(x), axis = 1)

            # One-hot encode true label
            y_one_hot = np.zeros((np.size(self.W, 0),1))
            y_one_hot[y] = 1

            # Softmax function that gives the label probabilities according to the model
            label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
            
            # SGD update
            self.update_weight(x, y_one_hot, label_probabilities)

    def predict(self, inputs):
        """
        inputs (n_examples, n_features + 1): observations
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        y_hat = np.argmax(inputs.dot(self.W.T), axis = 1)
        return y_hat

    def evaluate(self, inputs, labels):
        """
        inputs (n_examples, n_features + 1): observations
        y (n_examples,): gold labels

        returns classifier accuracy
        """
        y_hat = self.predict(inputs)
        accuracy = np.mean(y_hat == labels)
        return accuracy


def compute_hog_features(X, img_size=28, cells_per_dim=4, num_bins=9, progress_step=5000):
    """
    Compute simplified HOG features for a dataset of flattened images.
    HOG -> Histograms of Oriented Gradients

    Parameters:
        X (n_examples, n_features): original observations
        img_size (int): width/height of the original square image
        cells_per_dim (int): number of cells along each dimension
        num_bins (int): number of orientation histogram bins
    Returns:
        X_hog (n_examples, cells_per_dim*cells_per_dim*num_bins): observations in the new feature representation
    """
    n_examples = X.shape[0]
    cell_size = img_size // cells_per_dim  # e.g. 7x7 cells
    X_hog = np.zeros((n_examples, cells_per_dim * cells_per_dim * num_bins))

    # Bin edges for orientations in degrees (signed gradients)
    bin_edges = np.linspace(-180, 180, num_bins + 1)

    start = time.time()
    print(f'[HOG] Starting feature extraction on {n_examples} images...')

    for idx in range(n_examples):
        img = X[idx].reshape(img_size, img_size)

        # Compute gradients using finite differences
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)

        gx[:, :-1] = img[:, 1:] - img[:, :-1]   # horizontal gradient (kernel [ -1, +1 ])
        gy[:-1, :] = img[1:, :] - img[:-1, :]   # vertical gradient   (kernel [ -1, +1 ]^T)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.degrees(np.arctan2(gy, gx))  # angles in degrees

        hog_features = []

        # Loop over cells
        for i in range(cells_per_dim):
            for j in range(cells_per_dim):
                cell_mag = magnitude[
                    i*cell_size:(i+1)*cell_size,   # Get the block of pixels in this cell
                    j*cell_size:(j+1)*cell_size
                ]
                cell_ori = orientation[
                    i*cell_size:(i+1)*cell_size,
                    j*cell_size:(j+1)*cell_size
                ]

                hist, _ = np.histogram(
                    cell_ori,
                    bins=bin_edges,
                    weights=cell_mag,
                    density=False
                )

                hog_features.extend(hist)   # Concatenate histograms in a single feature vector

        X_hog[idx] = np.array(hog_features)

        # Print progress
        if (idx + 1) % progress_step == 0 or idx == n_examples - 1:
            percent = (idx + 1) / n_examples * 100
            print(f'[HOG] Processed {idx+1}/{n_examples} images ({percent:.1f}%)')

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('[HOG] Computation took {} minutes and {} seconds'.format(minutes, seconds))

    print(f'[HOG] Done. Final feature matrix shape: {X_hog.shape}')
    return X_hog

def grid_search_logistic(y_train, y_valid, y_test,
                         feature_names, features_dict,
                         learning_rates, regularizations,
                         epochs, save_path):
    """
    Perform grid search over learning rates, L2 penalties, and feature representations.

    Parameters:
        y_train, y_valid, y_test (n_examples,): arrays of target values for training, validation, and test data
        feature_names (list of str): Names of the feature representations
        features_dict (dict): Maps feature name to the corresponding data tuple: (X_train, X_valid, X_test)
        learning_rates (list of float): Candidate learning rates
        regularizations (list of float): Candidate L2 penalty values
        epochs (int): Number of epochs to train each configuration
        save_path (str): Temporary path to save checkpoints
    Returns:
        results (dict): Nested dictionary with validation accuracies for each configuration,
                        best test accuracy, and other info.
    """

    results = {}
    best_valid = -1
    best_feat = None
    best_config = None
    epochs = np.arange(1, args.epochs + 1)

    for feat_name in feature_names:
        X_tr, X_val, X_te = features_dict[feat_name]
        for lr, reg in itertools.product(learning_rates, regularizations):
            config_name = f"{feat_name}_lr{lr}_reg{reg}"
            print(f"\n[GRID] Training configuration: {config_name}")

            n_classes = np.unique(y_train).size
            n_feats = X_tr.shape[1]

            model = LogisticRegression(n_classes=n_classes,
                                       n_features=n_feats,
                                       eta=lr,              # learning rate
                                       regularization=reg)  # L2 penalty

            train_accs, valid_accs = [], []
            start = time.time()
            best_val_acc = -1
            best_epoch = -1

            for epoch in epochs:
                print('Training epoch {}'.format(epoch))
                # Shuffle training data
                train_order = np.random.permutation(X_tr.shape[0])
                X_tr_epoch = X_tr[train_order]
                y_tr_epoch = y_train[train_order]

                model.train_epoch(X_tr_epoch, y_tr_epoch)

                train_acc = model.evaluate(X_tr, y_train)
                val_acc = model.evaluate(X_val, y_valid)

                train_accs.append(train_acc)
                valid_accs.append(val_acc)

                print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, val_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch

            elapsed_time = time.time() - start
            results[config_name] = {
                "best_valid": float(best_val_acc),
                "selected_epoch": int(best_epoch),
                "time": elapsed_time
            }

            # Check if this is the overall best
            if best_val_acc > best_valid:
                best_valid = best_val_acc
                best_config = config_name
                best_feat = feat_name
                model.save(save_path)

    print("\nReloading best checkpoint")
    best_model = LogisticRegression.load(save_path)

    # Evaluate best model on test set
    X_test = features_dict[best_feat][2]
    test_acc = best_model.evaluate(X_test, y_test)
    results["best_config"] = best_config
    results["best_test_acc"] = float(test_acc)

    print(f"\n[GRID] Best configuration: {best_config}, Test accuracy: {test_acc:.4f}")
    return results

def load_data(args, mode):
    """
    Loads dataset with bias handling depending on the mode.
    """
    add_bias = (mode == "single")  # only single model expects bias=True
    data = utils.load_dataset(data_path=args.data_path, bias=add_bias)
    return data

def run_single_experiment(args, data):
    """
    Implements Q1.2(a): train a single logistic regression model.
    """
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = LogisticRegression(
        n_classes=n_classes,
        n_features=n_feats,
        eta=0.0001,             # learning rate
        regularization=0.00001  # L2 penalty
    )

    epochs = np.arange(1, args.epochs + 1)
    train_accs, valid_accs = [], []
    best_valid = 0.0
    best_epoch = -1

    start = time.time()

    for epoch in epochs:
        print('Training epoch {}'.format(epoch))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # save the best model checkpoint
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = epoch
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = LogisticRegression.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f, indent=4
        )

def run_grid_search_experiment(args, data):
    """
    Implements Q1.2(c): HOG features + grid search.
    """
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    X_train_hog = compute_hog_features(X_train)
    X_valid_hog = compute_hog_features(X_valid)
    X_test_hog = compute_hog_features(X_test)

    # Add bias term
    X_train_raw = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_valid_raw = np.hstack([X_valid, np.ones((X_valid.shape[0], 1))])
    X_test_raw  = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_train_hog = np.hstack([X_train_hog, np.ones((X_train_hog.shape[0], 1))])
    X_valid_hog = np.hstack([X_valid_hog, np.ones((X_valid_hog.shape[0], 1))])
    X_test_hog  = np.hstack([X_test_hog, np.ones((X_test_hog.shape[0], 1))])

    features_dict = {
        "hog": (X_train_hog, X_valid_hog, X_test_hog),
        "raw": (X_train_raw, X_valid_raw, X_test_raw)
    }

    learning_rates = [0.0001, 0.001, 0.01]
    regularizations = [0.00001, 0.0001]
    feature_names = ["hog", "raw"]

    results = grid_search_logistic(
        y_train=y_train, y_valid=y_valid, y_test=y_test,
        feature_names=feature_names,
        features_dict=features_dict,
        learning_rates=learning_rates,
        regularizations=regularizations,
        epochs=args.epochs,
        save_path=args.save_path
    )

    # Save results
    with open(args.scores, "w") as f:
        json.dump(results, f, indent=4)


def main(args):
    utils.configure_seed(seed=args.seed)
    data = load_data(args, mode=args.mode)

    if args.mode == "single":
        run_single_experiment(args, data)
    elif args.mode == "grid":
        run_grid_search_experiment(args, data)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'grid'], required=True,
                        help="Choose experiment: 'single' for 1.2(a), 'grid' for 1.2(c)")
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-logistic-regression-accs.pdf")
    parser.add_argument("--scores", default="Q1-logistic-regression-scores.json")
    args = parser.parse_args()
    main(args)