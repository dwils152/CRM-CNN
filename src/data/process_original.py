import numpy as np
from Bio import SeqIO
import sys
from multiprocessing import Pool

def one_hot_encode(seq, length):
    # Trim the sequence if it is too long
    if len(seq) > length:
        seq = seq[:length]

    encoded_seq = np.zeros(shape=(4, length))

    # Compute the start position for the sequence in the encoded sequence
    start_pos = (length - len(seq)) // 2

    for nt in range(len(seq)): 
        if seq[nt] == 'A':
            encoded_seq[:, start_pos + nt] = [1, 0, 0, 0]
        elif seq[nt] == 'C':
            encoded_seq[:, start_pos + nt] = [0, 1, 0, 0]
        elif seq[nt] == 'G':
            encoded_seq[:, start_pos + nt] = [0, 0, 1, 0]
        elif seq[nt] == 'T':
            encoded_seq[:, start_pos + nt] = [0, 0, 0, 1]
        elif seq[nt] == 'N':
            encoded_seq[:, start_pos + nt] = [0, 0, 0, 0]

    return encoded_seq

def process_sequence(args):
    i, record, length = args
    encoded = one_hot_encode(record.seq, length)
    return i, encoded

def mmap(pos_fasta, neg_fasta, length):

    #upper case the sequences
    pos = [seq.upper() for seq in SeqIO.parse(pos_fasta, 'fasta')]
    neg = [seq.upper() for seq in SeqIO.parse(neg_fasta, 'fasta')]

    num_samples = len(pos) + len(neg)
    seqs = np.memmap('SEQS.npy', dtype='float32', mode='w+', shape=(num_samples*4, length))
    labels = np.memmap('LABELS.npy', dtype='float32', mode='w+', shape=(num_samples, 1))
    
    # process positive sequences
    for i, record in enumerate(pos):
        idx, encoded = process_sequence((i, record, length))
        labels[idx] = 1
        seqs[idx*4:(idx+1)*4] = encoded

    # process negative sequences
    for i, record in enumerate(neg, start=len(pos)):
        idx, encoded = process_sequence((i, record, length))
        labels[idx] = 0
        seqs[idx*4:(idx+1)*4] = encoded


if __name__ == '__main__':
    mmap(sys.argv[1], sys.argv[2], int(sys.argv[3]))
