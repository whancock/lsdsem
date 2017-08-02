""" run the stanford parser
    the stanford pipeline requires raw text, one sentence per line
"""

import os
import glob
import json
import subprocess

from os.path import join, isfile, basename


SCRIPT_DIR = '/home/william.hancock/workspace/lib/stanford-corenlp-full-2017-06-09'

TMP_DIR = '/home/william.hancock/workspace/data/lsdsem/tmp'
# INPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/endings_raw_txt'
OUTPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/tmp/stanford'


# change to the directory of our AMR scripts
os.chdir(SCRIPT_DIR)

CLASSPATH = os.path.join(SCRIPT_DIR, '*')
PIPELINE = 'edu.stanford.nlp.sentiment.SentimentPipeline'


END_ONE = 'col_1'
END_TWO = 'col_2'


SENT_MAPPING = {
    'Verypositive': 4,
    'Positive': 3,
    'Neutral': 2,
    'Negative': 1,
    'Verynegative': 0
}


def extract_feats(files):

    """
    take a list of file that have been output by the stanford corenlp parser
    return a list of lists; each list is a vector of features
    E.G. one list will be the sentiment of dev_ending_1 
    """

    feats = []

    for meta, fname in files:

        m_list = list(meta)

        with open(fname) as parse_file:
            data = json.loads(parse_file.read())
            sent_feats = [SENT_MAPPING[sf['sentiment']] for sf in data['sentences']]
            feats.append(( tuple(m_list + ['sent']) , sent_feats))

    return feats





def parse(text_files):


    files_exist = True


    # get a list of the files corenlp just output
    # corenlp just adds a .json extension to the original input filenames
    # so we just recreate that here
    parse_files = []

    for meta, fname in text_files:

        out_xform = join(OUTPUT_DIR, basename(fname)) + '.json'
        parse_files.append((meta, out_xform))

        if not isfile(out_xform):
            files_exist = False



    if not files_exist:

        # create a list of the files we just created and write to a file
        # as stanford corenlp takes this as an arg and is faster this way
        file_list_path = create_file_list(text_files)

        subprocess.run([
            'java', 
            '-mx20g', 
            '-cp', CLASSPATH, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', 
            '-annotators', 'tokenize,ssplit,pos,parse,sentiment', 
            '-ssplit.eolonly', 'true',
            '-filelist', file_list_path, 
            '-outputDirectory', OUTPUT_DIR, 
            '-outputFormat', 'json'])
    


    # extract the features that we want and write to out_dir
    return extract_feats(parse_files)

    
    


def sentiment(file_list_path):

    pass

    # files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))

    # for fname in files:

        # for outputting stanford parser stuff


        # for outputting sentiment
        # base = os.path.splitext(os.path.basename(fname))[0]
        # out_file = base + '.sent.txt'

        # with open(os.path.join(OUTPUT_DIR, out_file), mode='w') as out_path:
        #     subprocess.run(['java', '-cp', CLASSPATH, 'edu.stanford.nlp.sentiment.SentimentPipeline', '-file', fname, '-output', 'root'], stdout=out_path)




def create_file_list(file_list):

    fpath = join(TMP_DIR, 'file_list.txt')

    with open(fpath, mode='w') as f:

        for _, fname in file_list:
            f.write(fname + '\n')

    return fpath