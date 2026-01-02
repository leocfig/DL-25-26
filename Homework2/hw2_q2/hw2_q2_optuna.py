import json
import os
import optuna
import torch
from torch.utils.data import DataLoader
import argparse

from config import RNAConfig, RNNHyperparamSpace, CNNHyperparamSpace
from utils import (
    load_rnacompete_data,
    configure_seed,
    reshape_tensor_dataset,
    subset_dataset
)

from rnn_model import RNN, train_epoch as train_epoch_rnn, evaluate as evaluate_rnn
from cnn_model import CNN, train_epoch as train_epoch_cnn, evaluate as evaluate_cnn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def objective_rnn(trial):
    print(f"=== Trial {trial.number} : Hyperparameters ===")
    print("Model: RNN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    space = RNNHyperparamSpace()

    hidden_size = trial.suggest_categorical("hidden_size", space.hidden_size)
    batch_size = trial.suggest_categorical("batch_size", space.batch_size)
    learning_rate = trial.suggest_float("lr", space.lr_min, space.lr_max, log=True)
    bidirectional = trial.suggest_categorical("bidirectional", space.bidirectional_options)
    dropout = trial.suggest_float("dropout", space.dropout_min, space.dropout_max)

    print(f"hidden_size: {hidden_size}, batch_size: {batch_size}, lr: {learning_rate:.5f}\n")
    configure_seed(RNAConfig.SEED)

    # Load full datasets
    full_train_ds = load_rnacompete_data("RBFOX1", split="train")
    full_val_ds = load_rnacompete_data("RBFOX1", split="val")

    # Subsample
    train_ds = subset_dataset(
        full_train_ds, fraction=RNAConfig.DATA_FRACTION, seed=RNAConfig.SEED
    )
    val_ds = subset_dataset(
        full_val_ds, fraction=RNAConfig.DATA_FRACTION, seed=RNAConfig.SEED
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    model = RNN(
        input_size=4,
        hidden_size=hidden_size,
        output_size=1,
        bidirectional=bidirectional,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ---- Training + Evaluation ----
    best_val_spearman = -1.0

    for epoch in range(1, space.num_epochs + 1):
        train_loss = train_epoch_rnn(train_loader, model, optimizer, device)
        val_spearman = evaluate_rnn(val_loader, model, device)
        best_val_spearman = max(best_val_spearman, val_spearman)
        print(f"Epoch {epoch}/{space.num_epochs} | Train Loss: {train_loss:.4f} | Val Spearman: {val_spearman:.4f}")


    return best_val_spearman


def objective_cnn(trial):
    print(f"=== Trial {trial.number} : Hyperparameters ===")
    print("Model: CNN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    space = CNNHyperparamSpace()

    conv_params = trial.suggest_categorical(
        "conv_params", space.conv_params
    )
    fc_params = trial.suggest_categorical(
        "fc_params", space.fc_params
    )
    batch_size = trial.suggest_categorical(
        "batch_size", space.batch_size
    )
    learning_rate = trial.suggest_float(
        "lr", space.lr_min, space.lr_max, log=True
    )

    print(f"conv_params: {conv_params}, fc_params: {fc_params}, batch_size: {batch_size}, lr: {learning_rate:.5f}\n")
    configure_seed(RNAConfig.SEED)

    # ------ Loading + Reshaping Data ------
    # train_ds = reshape_tensor_dataset(
    #     load_rnacompete_data("RBFOX1", "train")
    # )
    # val_ds = reshape_tensor_dataset(
    #     load_rnacompete_data("RBFOX1", "val")
    # )

    # train_loader = DataLoader(
    #     train_ds, batch_size=batch_size, shuffle=True
    # )
    # val_loader = DataLoader(
    #     val_ds, batch_size=batch_size, shuffle=False
    # )

    full_train_ds = reshape_tensor_dataset(
        load_rnacompete_data("RBFOX1", "train")
    )
    full_val_ds = reshape_tensor_dataset(
        load_rnacompete_data("RBFOX1", "val")
    )

    train_ds = subset_dataset(
        full_train_ds, fraction=RNAConfig.DATA_FRACTION, seed=RNAConfig.SEED
    )
    val_ds = subset_dataset(
        full_val_ds, fraction=RNAConfig.DATA_FRACTION, seed=RNAConfig.SEED
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    # ---- Model ----
    model = CNN(
        conv_params=conv_params,
        fc_params=fc_params,
        input_size=(1, RNAConfig.SEQ_MAX_LEN, 4)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    # ---- Training + Evaluation ----
    best_val_spearman = -1.0

    for epoch in range(1, space.num_epochs + 1):
        train_loss = train_epoch_cnn(train_loader, model, optimizer)
        val_spearman = evaluate_cnn(val_loader, model)
        best_val_spearman = max(best_val_spearman, val_spearman)
        print(f"Epoch {epoch}/{space.num_epochs} | Train Loss: {train_loss:.4f} | Val Spearman: {val_spearman:.4f}")

    return best_val_spearman


def run_rnn_study():
    study = optuna.create_study(direction="maximize",
                                study_name="RNN_Optuna_Study",
                                sampler=optuna.samplers.TPESampler(seed=RNAConfig.SEED))

    study.optimize(objective_rnn, n_trials=30)

    best = {
        "best_params": study.best_params,
        "best_spearman": study.best_value
    }

    os.makedirs("optuna_results", exist_ok=True)
    with open("optuna_results/best_rnn_params.json", "w") as f:
        json.dump(best, f, indent=4)

    print("Best RNN params:", study.best_params)
    print("Best RNN Spearman:", study.best_value)


def run_cnn_study():
    study = optuna.create_study(direction="maximize",
                                study_name="CNN_Optuna_Study",
                                sampler=optuna.samplers.TPESampler(seed=RNAConfig.SEED))

    study.optimize(objective_cnn, n_trials=30)

    best = {
        "best_params": study.best_params,
        "best_spearman": study.best_value
    }

    os.makedirs("optuna_results", exist_ok=True)
    with open("optuna_results/best_cnn_params.json", "w") as f:
        json.dump(best, f, indent=4)

    print("Best CNN params:", study.best_params)
    print("Best CNN Spearman:", study.best_value)


def main(opt):
    if opt.model == "rnn":
        run_rnn_study()

    elif opt.model == "cnn":
        run_cnn_study()

    elif opt.model == "both":
        run_rnn_study()
        run_cnn_study()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for RNA binding models"
    )

    parser.add_argument('-model', default="both", type=str,
                        choices=["rnn", "cnn", "both"],
                        help="""Which model to optimize: rnn, cnn, or both.""")

    opt = parser.parse_args()
    main(opt)
