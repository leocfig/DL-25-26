#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
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


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
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

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
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
            best_epoch = i
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
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-logistic-regression-accs.pdf")
    parser.add_argument("--scores", default="Q1-logistic-regression-scores.json")
    args = parser.parse_args()
    main(args)
