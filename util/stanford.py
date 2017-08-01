""" run the JAMR parser 
    expects data to be in a specific format; see split_tokenize.py
"""

import os
import glob
import subprocess


SCRIPT_DIR = '/home/william.hancock/workspace/lib/stanford-corenlp-full-2017-06-09'

INPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/endings_raw_txt'
OUTPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/stanford'


# change to the directory of our AMR scripts
os.chdir(SCRIPT_DIR)

CLASSPATH = os.path.join(SCRIPT_DIR, '*')
PIPELINE = 'edu.stanford.nlp.sentiment.SentimentPipeline'

# get a list of our input files
files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))

for fname in files:

    # for outputting stanford parser stuff
    subprocess.run(['./corenlp.sh', '-annotators tokenize,ssplit,pos,parse,sentiment', '-file', fname, '-outputDirectory', OUTPUT_DIR, '-outputFormat', 'json'])


    # for outputting sentiment
    # base = os.path.splitext(os.path.basename(fname))[0]
    # out_file = base + '.sent.txt'

    # with open(os.path.join(OUTPUT_DIR, out_file), mode='w') as out_path:
    #     subprocess.run(['java', '-cp', CLASSPATH, 'edu.stanford.nlp.sentiment.SentimentPipeline', '-file', fname, '-output', 'root'], stdout=out_path)



