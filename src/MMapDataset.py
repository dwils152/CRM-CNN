import torch
from torch.utils.data import Dataset
import numpy as np
from Bio import SeqIO

class MMapDataset(Dataset):
    def __init__(self, seqs_path, labels_path,  fasta, length, use_annotations=False):
        
        self.fasta = fasta
        self.labels_map = np.memmap(labels_path, dtype='float32', mode='r', shape=(self._dataset_size(), 1))
        self.seqs_map = np.memmap(seqs_path, dtype='float32', mode='r', shape=(self._dataset_size()*4, length))

        self.use_annotations = use_annotations
        self.annotations = []
        if use_annotations:
            record_ids = [record.id for record in SeqIO.parse(fasta, 'fasta')]

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
    
def debug():
    dataset = MMapDataset('/projects/zcsu_research1/dwils152/Motif-Cnn/Data-v2/mm10_100_0_None_0.5/SEQS.npy',
                          '/projects/zcsu_research1/dwils152/Motif-Cnn/Data-v2/mm10_100_0_None_0.5/LABELS.npy',
                          '/projects/zcsu_research1/dwils152/Motif-Cnn/Data-v2/mm10_100_0_None_0.5/pos_neg.fa',
                          100,
                          use_annotations=True)

    seq, label, annot = dataset[0]
    print(seq, label)
    
    print(dataset._dataset_size())


    #labels = []
    #datase_size = dataset._dataset_size()

    #for i in range(len(datase_size)):
    #    labels.append(dataset.__getitem__(i))

    #print(set(labels))
    
    #print(dataset.length)
    
#debug()
    
                          
