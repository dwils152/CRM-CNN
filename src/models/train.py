import argparse
from utils import get_data_loader
import LightningCrmCNN as LightningCrmCNN 
from CrmCNN import CrmCNN
from MMapDataset import MMapDataset
from torch.utils.data import DataLoader, random_split
from torch import Generator
import pytorch_lightning as pl

def main(args):

    data = MMapDataset(seqs_path=args.seqs,
                       labels_path=args.labels, 
                       fasta_path=args.fasta,
                       length=args.length,
                       use_annotations=False)

    splits = (random_split(data, [0.6, 0.2, 0.2], generator=Generator().manual_seed(42)))
    train_loader = get_data_loader(splits[0], batch_size=32)
    val_loader = get_data_loader(splits[1], batch_size=32)
    test_loader = get_data_loader(splits[2], batch_size=32)
    
    model = LightningCrmCNN(CrmCNN())
    trainer = pl.Trainer(accelerator="gpu", devices=4, num_nodes=1, strategy="ddp")
    trainer.fit(model, train_loader, val_loader)
    

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
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1, 
                        help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)
