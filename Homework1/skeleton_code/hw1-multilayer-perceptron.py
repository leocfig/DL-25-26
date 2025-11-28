#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt
import utils

def relu(x):
    return np.clip(x, 0, None)

def relu_derivative(x):
    return x > 0

def softmax(z, axis=None):
    z -= np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


def cross_entropy_loss(pred, y_true):
    return -np.log(pred[y_true] + 1e-9)

class MultilayerPerceptron:
    """
    x ---------> z[1] ---relu---> h[1] --------> z[2] --softmax--> out
    hidden layer activation function: relu
    output layer: softmax (classification problem)
    """
    def __init__(self, n_features, n_classes,  hidden_dim, eta):
        self.eta = eta
        self.n_classes = n_classes
        # Weight initialization: N(0.1, 0.1^2)
        in_sizes = [n_features] + [hidden_dim]
        out_sizes = [hidden_dim] + [n_classes]
        self.weights = [np.random.normal(size=(in_size, out_size),
                                         loc=0.1, scale=0.1)
                        for in_size, out_size in zip(in_sizes, out_sizes)]
        self.biases = [np.zeros(out_size) for out_size in out_sizes]
        self.activations = [relu] + [softmax]
        self.epsilon = 1e-6

    def save(self, path):
        """ Save model (pickle entire object). """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """ Load model from path. """
        with open(path, "rb") as f:
            return pickle.load(f)

    def forward_propagation(self, x):
        hiddens = [x]   # h_0 = input
        z_vals = []     # store pre-activations z_i

        for W, activation, b in zip(self.weights, self.activations, self.biases):
            z_i = np.dot(hiddens[-1], W) + b   # pre-activation
            z_vals.append(z_i)
            h_i = activation(z_i)              # post-activation
            hiddens.append(h_i)

        return z_vals, hiddens

    def back_propagation(self, z_vals, hiddens, y_true):
        L = len(self.weights)
        z_grads = [None] * L
        h_grads = [None] * L
        w_grads = [None] * L
        b_grads = [None] * L

        # output layer gradient
        probs = hiddens[-1]              # final softmax output
        #loss = -np.log(probs[y_true] + self.epsilon)
        loss = cross_entropy_loss(probs, y_true)

        out_grad = probs.copy()          # softmax derivative trick
        out_grad[y_true] -= 1
        z_grads[-1] = out_grad           # dL/dz_L

        for i in reversed(range(L)):
            w_grads[i] = np.outer(hiddens[i], z_grads[i])
            b_grads[i] = z_grads[i]

            if i > 0:
                # Gradient wrt hidden unit outputs
                h_grads[i] = z_grads[i] @ self.weights[i].T

                # Gradient wrt z at previous layer
                z_grads[i-1] = h_grads[i] * relu_derivative(z_vals[i-1])

        return w_grads, b_grads, loss

    def update_weights(self, w_grads, b_grads):
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * w_grads[i]
            self.biases[i]  -= self.eta * b_grads[i]
        
    def train_epoch(self, inputs, labels):
        loss = 0
        for x_i, y_i in zip(inputs, labels):
            z, hiddens = self.forward_propagation(x_i)

            w_grads, b_grads, sample_loss = self.back_propagation(z, hiddens, y_i)

            loss += sample_loss
            self.update_weights(w_grads, b_grads) # stochastic gradient descent
        
        return loss / inputs.shape[0]

    def predict(self, inputs):
        preds = []
        for x in inputs:
            _, hiddens = self.forward_propagation(x)
            probs = hiddens[-1]          # softmax output
            preds.append(probs)
        return np.array(preds).argmax(axis=1)
    
    def evaluate(self, x, y):
        y_hat = self.predict(x)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)

    X_train, y_train = data["train"]
    X_valid, y_valid   = data["dev"]
    X_test, y_test  = data["test"]

    n_features = X_train.shape[1]
    n_classes = np.unique(y_train).size
    hidden_dim = 100

    model = MultilayerPerceptron(n_features, n_classes, hidden_dim, eta=0.001)
    print('N features: {}'.format(n_features))
    print('N classes: {}'.format(n_classes))

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []
    train_losses = []

    start = time.time()
    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))

        # Shuffle training set each epoch
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        # Train one epoch
        epoch_loss = model.train_epoch(X_train, y_train)
        train_losses.append(epoch_loss)

        # Evaluate
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f} | loss: {:.4f}'.format(train_acc, valid_acc, epoch_loss))

        # Save best checkpoint (pickle the whole model)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)
    
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = MultilayerPerceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot("Epoch", "Loss", {
        "train Loss": (epochs, train_losses)
    }, filename=args.loss_plot)

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    # Save scores json
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
                        help="Number of epochs to train for.")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--accuracy-plot', default="Q1-mlp-training.pdf",
                        help="File to save the combined training plot (loss + train/val accuracy).")
    parser.add_argument('--loss-plot', default="Q1-mlp-loss.pdf",
                        help="File to save the training loss.")
    parser.add_argument('--scores', default="Q1-mlp-scores.json",
                        help="JSON file to save best epoch and accuracies.")
    args = parser.parse_args()
    main(args)
