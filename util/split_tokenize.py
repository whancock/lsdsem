"""
the jamr parser takes in tokenized sentences one sent per line
this util file reads in our corpus data form csv files
and outputs the desired format for jamr
"""

from os.path import join
import csv

import nltk



CORPUS_ROOT = '/home/william.hancock/workspace/data/lsdsem/raw'
TARGET_FILE = 'cloze_test_val_spring_2016.csv'

OUT_DIR = '/home/william.hancock/workspace/data/lsdsem/endings_raw_txt'
TOKENIZED_DIR = '/home/william.hancock/workspace/data/lsdsem/tok_split'

def main():

    cols = None

    with open(join(CORPUS_ROOT, TARGET_FILE), newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:

            if not cols:
                cols = [[] for x in range(len(row))]

            for (idx, item) in enumerate(row):
                cols[idx].append(item)


    target_cols = cols[1:7]

    in_file = TARGET_FILE.split('.')

    for (idx, col) in enumerate(target_cols):

        out_file = in_file[0] + '_col_' + str(idx) + '.txt'

        with open(join(OUT_DIR, out_file), mode='w') as f:
            for sent in col:
                f.write(sent + '\n')

        # tk_out_file
        # tkn_sent = nltk.word_tokenize(sent)
        # f.write(' '.join(tkn_sent) + '\n')

    print('complete')






if __name__ == "__main__":
    # execute only if run as a script
    main()