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

def objective(trial):
    
    data = MMapDataset(seqs_path=args.seqs,
                    labels_path=args.labels, 
                    fasta_path=args.fasta,
                    length=args.length,
                    use_annotations=False)

    splits = (random_split(data, [0.6, 0.2, 0.2], generator=Generator().manual_seed(42)))
    train_loader = get_data_loader(splits[0], batch_size=32)
    val_loader = get_data_loader(splits[1], batch_size=32)
    test_loader = get_data_loader(splits[2], batch_size=32)
    
    config = {
        "batch_size": 16,
        "conv_stride": 1,
        "pool_stride": 1,
        "pool_size": 3,
        "dropout": 0.25,
        "kernels": 100,
        "layer_n_kernels": 5,
        "kernel_len": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "momentum": 0.95,
        "dilation": 1,
        "num_layers": 2,
        "act_fn": 'leaky_relu',
        "training_data_len": 200
    }
    
    model = LightningCrmCNN(CrmCNN(config))
    trainer = pl.Trainer(max_epochs=args.epochs,
                        accelerator='gpu',
                        devices=args.gpus,
                        num_nodes=1,
                        strategy='ddp',
                        logger=True)
    
    trainer.fit(model, train_loader, val_loader)

def main(args):
     
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    
    trial = study.best_trial
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN with PyTorch Lightning")
    
    parser.add_argument("--seqs", type=str, default="../../data/processed/SEQS_200.npy",
                        help="Path to the sequences")
    parser.add_argument("--labels", type=str, default="../../data/processed/LABELS_200.npy",
                        help="Path to the labels")
    parser.add_argument("--fasta", type=str, default="../../data/processed/pos_neg_200.fa",
                        help="Path to the fasta file")
    parser.add_argument("--length", type=int, default=200, 
                        help="Length of the sequences")
    parser.add_argument("--use_annotations", type=bool, default=False, 
                        help="Whether to use annotations")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=4, 
                        help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)
