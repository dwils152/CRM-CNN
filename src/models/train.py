import argparse
from utils import get_data_loader
from LightningCrmCNN import LightningCrmCNN
from CrmCNN import CrmCNN
from MMapDataset import MMapDataset
from torch.utils.data import DataLoader, random_split
from torch import Generator
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import random
import numpy as np
import torch

def objective(trial):
    
    data = MMapDataset(seqs_path=args.seqs,
                    labels_path=args.labels, 
                    fasta_path=args.fasta,
                    length=args.length,
                    use_annotations=False)
    
    config = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024]),
        "conv_stride": trial.suggest_int("conv_stride", 1, 1),
        "pool_stride": trial.suggest_int("pool_stride", 1, 1),
        "pool_size": trial.suggest_int("pool_size", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.7, step=0.05),
        "kernels": trial.suggest_int("kernels", 50, 100),
        "layer_n_kernels": trial.suggest_int("layer_n_kernels", 50, 100),
        "kernel_len": trial.suggest_int("kernel_len", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.0, 1.0, step=0.05),
        "dilation": trial.suggest_int("dilation", 1, 1),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "act_fn": trial.suggest_categorical("act_fn", ["relu"]),
        "training_data_len": 200
    }
    
    splits = (random_split(data, [0.6, 0.2, 0.2], generator=Generator().manual_seed(42)))
    train_loader = get_data_loader(splits[0], batch_size=config["batch_size"])
    val_loader = get_data_loader(splits[1], batch_size=config["batch_size"])
    test_loader = get_data_loader(splits[2], batch_size=config["batch_size"])
    
    model = LightningCrmCNN(CrmCNN(config))
    
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    stopper = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=3)
    swa = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    
    trainer = pl.Trainer(max_epochs=args.epochs,
                        accelerator='auto',
                        devices=1,
                        num_nodes=1,
                        strategy='ddp_spawn',
                        logger=True,
                        precision=16,
                        callbacks=[pruner, stopper, swa])
    
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_fbeta"].item()


def main(args):
     
    storage_url = "mysql://root:my-secret-pw@192.168.170.242:3306/example"
    study = optuna.create_study(direction="maximize", storage=storage_url)
    study.optimize(objective, n_trials=10)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN with PyTorch Lightning")
    
    parser.add_argument("--seqs", type=str, default="../../data/processed/yeast/SEQS_200.npy",
                        help="Path to the sequences")
    parser.add_argument("--labels", type=str, default="../../data/processed/yeast/LABELS_200.npy",
                        help="Path to the labels")
    parser.add_argument("--fasta", type=str, default="../../data/processed/yeast/pos_neg_200.fa",
                        help="Path to the fasta file")
    parser.add_argument("--length", type=int, default=200, 
                        help="Length of the sequences")
    parser.add_argument("--use_annotations", type=bool, default=False, 
                        help="Whether to use annotations")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=4, 
                        help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)
