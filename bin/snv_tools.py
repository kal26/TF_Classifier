import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def bed_sort(file_path, out_path):
    """ Sort a bed file."""
    command = ['sort', '-k1,1', '-k2,2n', file_path]
    result = subprocess.run(command, stdout=open(out_path, 'w'), stderr=subprocess.PIPE, universal_newlines=True)
    return result.returncode, result.stdout, result.stderr

def homo_merge(snps, min_count):
    """ Homozygous merge of features with normalization and averaging.
    
    Arguments:
        snps -- loaded dataframe of snp bed file with columns [chr, start, end, refAllele, altAllele, refCount, altCount]
            (optionally, norm column)
        min_count -- number of read for an allele required to be included
    
    Reutrns:
        filtered_snps -- A combined dataframe with columns [chr, start, end, refAllele, allele, ref_count, alt_count]
    """
    ref_row = snps.iloc[0]
    allele_to_n = {'a':0, 'c':1, 't':2, 'g':3}
    n_to_allele = ['a', 'c', 't', 'g']
    # count is the number of reads in a, c, t, g
    count = [0,0,0,0]
    # exp_count is the number of contributing experiments
    exp_count = [0,0,0,0]
    filtered_snps = list()

    # test for norms
    norms=True
    try:
        a = snps.norm
    except AttributeError:
        print('SNPs had no norm attribute.')
        norms=False

    def update_count(count, exp_count, row):
        """ Generate an array from snps row."""
        if (int(row.refCount) > min_count):
            if norms:
                 count[allele_to_n[row.refAllele.lower()]] += int(row.refCount)/row.norm
            else:
                 count[allele_to_n[row.refAllele.lower()]] += int(row.refCount)
            exp_count[allele_to_n[row.refAllele.lower()]] += 1
        if (int(row.altCount) > min_count):
            if norms:
                count[allele_to_n[row.altAllele.lower()]] += int(row.altCount)/row.norm
            else:
                count[allele_to_n[row.altAllele.lower()]] += int(row.altCount)
            exp_count[allele_to_n[row.altAllele.lower()]] += 1
        return count, exp_count

    #cycle through
    for index, row in tqdm(snps.iterrows()):
        # at same position?
        if ref_row.start == int(row.start) and ref_row.chr == row.chr:
            update_count(count, exp_count, row)
        else:
            # write out the last bit of data
            for allele in n_to_allele:
                if ref_row['refAllele'].lower() == allele:
                    pass
                elif count[allele_to_n[allele]] > 0 and count[allele_to_n[ref_row.refAllele.lower()]] > 0:
                    # make a new row for each nonzero elem
                    ref_count = count[allele_to_n[ref_row.refAllele.lower()]] / exp_count[allele_to_n[ref_row.refAllele.lower()]]
                    alt_count = count[allele_to_n[allele]] / exp_count[allele_to_n[allele]]
                    new_row = [ref_row.chr, ref_row.start, ref_row.end, ref_row.refAllele.lower(), allele, ref_count, alt_count]
                    filtered_snps.append(new_row)
            # start for the next row
            ref_row = row
            count = [0, 0, 0, 0] 
            exp_count = [0, 0, 0, 0]
            update_count(count, exp_count, row)    

    # make the new dataframe
    columns=['chr', 'start', 'end', 'refAllele',  'altAllele',  'refCount',  'altCount']
    return pd.DataFrame(filtered_snps, columns=columns)

