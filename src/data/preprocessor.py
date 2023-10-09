import sys
import argparse
from Bio import SeqIO
import re

class ChromosomeSegmentor:
    def __init__(self, ref_genome, length=100, overlap=0, n_threshold=0.0):
        self.ref_genome = ref_genome
        self.ref_split = self.ref_genome.split(".")[0] + f"_{length}_{overlap}_{n_threshold}.fa"
        self.length = length
        self.overlap = overlap
        self.n_threshold = n_threshold

    def _pad_sequence(self, seq):
        pad_left = (self.length - len(seq)) // 2
        pad_right = self.length - len(seq) - pad_left
        return "N" * pad_left + seq + "N" * pad_right

    def _break_chromosome(self, chromosome, length=100, overlap=0):
        seq = str(chromosome.seq).upper()  # Convert sequence to uppercase
        chrom_segments = []
        for start in range(0, len(seq), length - overlap):
            end = start + length
            # Check to see of the end of the interval runs off the end of the chromosome
            if end <= len(seq):
                segment = seq[start:end]
            else:
                # If it does, pad the sequence with Ns
                segment = self._pad_sequence(seq[start:])
            n_count = segment.count("N")
            segment_length = len(segment)
            n_percent = n_count / segment_length
            if n_percent <= self.n_threshold:
                chrom_segments.append((start, segment))
        return chrom_segments

    def _break_chromosome_coverage(self, chromosome, length=100, overlap=0):
        seq = str(chromosome.seq).upper()  # Convert sequence to uppercase
        chrom_segments = []
        coords = []
        delim = r"[:-]"
        chrom, start, end = re.split(delim, chromosome.id)
        start, end = int(start), int(end)
        for st in range(start, end, length-overlap):
            en = st + length
            # Check to see of the end of the interval runs off the end of the chromosome
            if en <= end:
                segment = seq[st:en]
            else:
                # If it does, pad the sequence with Ns
                segment = self._pad_sequence(seq[st:])
            #n_count = segment.count("N")
            #segment_length = len(segment)
            #n_percent = n_count / segment_length
            #if n_percent <= self.n_threshold:
            chrom_segments.append((st, segment))
            coords.append((chrom, st, en))
        return coords, chrom_segments
    
    def process_fasta(self):
        with open(self.ref_split, "w") as fout:
            '''for record in SeqIO.parse(self.ref_genome, "fasta"):
                coords, chrom_segments = self._break_chromosome_coverage(record, self.length, self.overlap)
                for coord, (start, segment) in zip(coords, chrom_segments):
                    chrom, st, en = coord
                    fout.write(f">{chrom}:{start}-{en}\n{segment}\n")'''
                    
            for record in SeqIO.parse(self.ref_genome, "fasta"):
                chrom_segments = self._break_chromosome(record, self.length, self.overlap)
                for start, segment in chrom_segments:
                    fout.write(f">{record.id}:{start}-{start+len(segment)}\n{segment}\n")
        return self.ref_split
                
                


class IntervalGenerator:
    def __init__(self, split_genome, pred_interval_len=None):
        self.split_genome = split_genome
        self.pred_interval_len = pred_interval_len
        self.len, self.overlap, self.n_threshold = self.split_genome.split("_")[1:]
        self.org = args.ref_genome.split(".")[0]
        self.bed_intervals = f"{self.org}_intervals_{self.len}_{self.overlap}.bed"
        
    @staticmethod
    def _parse_id(id_string):
        """Helper method to parse sequence ID."""
        delim = r"[:-]"
        return re.split(delim, id_string)

    def _pred_context_intervals(self):
        fasta_sequences = SeqIO.parse(open(self.split_genome), 'fasta')
        with open("pred_intervals.bed", "w") as fout:
            with open("context_intervals.bed", "w") as fout2:
                id_num = 0
                for fasta in fasta_sequences:
                    chrom, start, end, *_ = self._parse_id(fasta.id)
                    midpoint = (int(start) + int(end)) // 2
                    pred_start = midpoint - (self.pred_interval_len // 2)
                    pred_end = midpoint + (self.pred_interval_len // 2)
                    fout.write(f"{chrom}\t{pred_start}\t{pred_end}\t{id_num}\n")
                    fout2.write(f"{chrom}\t{start}\t{end}\t{id_num}\n")
                    id_num += 1

    def _intervals(self):
        fasta_sequences = SeqIO.parse(open(self.split_genome), 'fasta')
        with open(self.bed_intervals, "w") as fout:
            id_num = 0
            for fasta in fasta_sequences:
                chrom, start, end, *_ = self._parse_id(fasta.id) 
                fout.write(f"{chrom}\t{start}\t{end}\t{id_num}\n")
                id_num += 1

    def process(self):
        if self.pred_interval_len is not None:
            self._pred_context_intervals()
        else:
            self._intervals()


def int_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value {value}. It should be 'None' or an integer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process fasta sequences.')
    parser.add_argument('ref_genome', type=str, help='Input fasta file')
    parser.add_argument('--length', type=int, default=100, help='Length of segments')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between segments')
    parser.add_argument('--pred_interval_len', type=int_or_none, default=None, help='Length of prediction interval')
    parser.add_argument('--n_threshold', type=float, default=0.0, help='Maximum fraction of N allowed in a segment')

    args = parser.parse_args()

    # Splits the genome into segments
    segmentor = ChromosomeSegmentor(args.ref_genome, args.length, args.overlap, args.n_threshold)
    split_genome = segmentor.process_fasta()

    # Reads the split genome and generates intervals
    #interval_generator = IntervalGenerator(split_genome, args.pred_interval_len)
    #interval_generator.process()

