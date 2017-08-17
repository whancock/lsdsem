""" process corpus and add our own features """

from collections import defaultdict


def len_diff(end_one, end_two):
    return len(end_one) - len(end_two)



DIFF_FUNCS = [
    ('lendiff', len_diff)
]


def end_len(ending):
    return len(ending)



SOLO_FUNCS = [
    ('len', end_len)
]



def parse(text_files):

    """
    take in a list of text files. in this case, we get 4 files: dev-end-1, dev-end-2, test-end-1, test-end-2
    for each, generate a set of features. each feature should be a tuple of (label, list(vals))
    """

    all_feats = {}

    for tset, files in text_files.items():

        feats = defaultdict(list)

        end_one_file = files['e1']
        end_two_file = files['e2']

        with open(end_one_file) as end_one, open(end_two_file) as end_two:

            for end_one, end_two in zip(end_one, end_two):

                end_one_split = end_one.split(' ')
                end_two_split = end_two.split(' ')

                for name, func in DIFF_FUNCS:
                    feats[name].append(func(end_one_split, end_two_split))

                for name, func in SOLO_FUNCS:
                    feats['e1_' + name].append(func(end_one))
                    feats['e2_' + name].append(func(end_two))

        all_feats[tset] = feats

    return all_feats