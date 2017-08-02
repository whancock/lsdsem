"""
the jamr parser takes in tokenized sentences one sent per line
this util file reads in our corpus data from csv files
and outputs the desired format for jamr
"""

from os.path import join, splitext, basename
import csv

import nltk


OUT_DIR = '/tmp'
CORPUS_ROOT = '/home/william.hancock/workspace/data/lsdsem/ukp'

def csv_to_text(files, tokenize=False):

    out_files = []

    for tag, fname in files:

        cols = None

        with open(fname, newline='') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')
            next(reader)

            for row in reader:

                if not cols:
                    cols = [[] for x in range(len(row))]

                for (idx, item) in enumerate(row):
                    cols[idx].append(item)


        target_cols = cols[5:7]

        in_file_split = splitext(basename(fname))

        out_ext = '.tok.txt' if tokenize else '.txt'

        for (idx, col) in enumerate(target_cols, start=1):

            out_file = in_file_split[0] + '_col_' + str(idx) + out_ext
            out_file_path = join(OUT_DIR, out_file)

            out_files.append(((tag, 'e'+str(idx)), out_file_path))

            with open(out_file_path, mode='w') as f:
                for sent in col:

                    if tokenize:
                        tkn_sent = nltk.word_tokenize(sent)
                        f.write(' '.join(tkn_sent) + '\n')
                    else:
                        f.write(sent + '\n')


    return out_files