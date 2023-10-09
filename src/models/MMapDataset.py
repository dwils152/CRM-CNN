import torch
from torch.utils.data import Dataset
import numpy as np
from Bio import SeqIO

class MMapDataset(Dataset):
    def __init__(self, seqs_path, labels_path,  fasta_path, length, use_annotations=False):
        
        self.fasta = fasta_path
        self.labels_map = np.memmap(labels_path, dtype='float32', mode='r', shape=(self._dataset_size(), 1))
        self.seqs_map = np.memmap(seqs_path, dtype='float32', mode='r', shape=(self._dataset_size()*4, length))

        self.use_annotations = use_annotations
        self.annotations = []
        if use_annotations:
            record_ids = [record.id for record in SeqIO.parse(fasta_path, 'fasta')]

            self.annotations = record_ids

    def _get_data(self, idx):
        s_idx = idx * 4 
        seq = np.array(self.seqs_map[s_idx:s_idx+4, :], dtype=np.float32)
        label = np.array([self.labels_map[idx]], dtype=np.float32)
        return torch.from_numpy(seq), (torch.from_numpy(label))

    def _dataset_size(self):
        with open(self.fasta, 'r') as fin:
            return sum(1 for line in fin if line.startswith('>'))

    def __getitem__(self, idx):
        seq, label = self._get_data(idx)
        if self.use_annotations:
            return seq, label, self.annotations[idx]
        return seq, label
    
    def __len__(self):
        return self._dataset_size()
    
    
                          
