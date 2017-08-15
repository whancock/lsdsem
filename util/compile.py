"""
compile all of our generated features together
write them to an augmented csv along with the original text

we will read this file with tensorflow
"""

import csv
from collections import defaultdict
from os.path import join, basename

import stanford
import features
from csv_to_text import csv_to_text


OUT_DIR = '/home/william.hancock/workspace/data/lsdsem/compiled'

FILES = [
    ('dev', '/home/william.hancock/workspace/data/lsdsem/ukp/dev_storycloze.csv'),
    ('test', '/home/william.hancock/workspace/data/lsdsem/ukp/test_storycloze.csv')]


FILE_TO_FEATS = {}

for label, _ in FILES:
    FILE_TO_FEATS[label] = []


def main():

    """ wrangle features from different sources and compile into csv files """

    # take the UKP files which are csv and extract the text we need into text files
    # so that we can run command lines parsers like stanford
    text_files = csv_to_text(FILES)

    tokenized_text_files = csv_to_text(FILES, tokenize=True)

    # run files through corenlp
    # stanford_feats = stanford.parse(text_files)
    # add_feats(stanford_feats)



    # for a lot of the features we want to calc, we need to do a comparison
    # between the two endings. tokenized_text_files is just a list of files, 
    # so we need to re-cluster them based on the metadata
    fset = defaultdict(dict)
    [fset[finfo[0][0]].update({finfo[0][1]: finfo[1]}) for finfo in tokenized_text_files]


    custom_feats = features.parse(fset)



    # take our features and write to OUT_DIR
    # write_feats()

    



def write_feats():

    """ augment existing csv files with our new features and output to new dir """

    for tag, fname in FILES:

        cur_feats = FILE_TO_FEATS[tag]
        out_path = join(OUT_DIR, basename(fname))

        with open(out_path, mode='w+') as out_file, open(fname, newline='') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(out_file, delimiter=',')

            header = next(reader)[:8]

            for meta, _ in cur_feats:
                header.append('_'.join(meta))

            writer.writerow(header)


            for idx, row in enumerate(reader):
                if row:
                    row_slice = row[:8]
                    for _, feat in cur_feats:
                        row_slice.append(feat[idx])
                    writer.writerow(row_slice)






def add_feats(feat_list):

    """ cluster features based on file """

    for meta, data in feat_list:
        FILE_TO_FEATS[meta[0]].append((meta[1:], data))




if __name__ == "__main__":
    # execute only if run as a script
    main()
