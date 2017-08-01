""" run the JAMR parser 
    expects data to be in a specific format; see split_tokenize.py
"""

import os
import glob
import subprocess

SCRIPT_DIR = '/home/william.hancock/workspace/jamr/'

INPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/tok_split'
OUTPUT_DIR = '/home/william.hancock/workspace/data/lsdsem/amr/'


# change to the directory of our AMR scripts
os.chdir(SCRIPT_DIR)


# get a list of our input files
files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))

for fname in files:

    # print(fname)
    out_name = ''

    subprocess.run(['./scripts/PARSE.sh', '<', fname, '>', out_name ])



