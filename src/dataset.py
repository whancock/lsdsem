""" reads a corpus """

from collections import Counter

import csv
from story import Story

class RocDataset:

    """ a class to help us use ROCStory data for machine learning """

    def __init__(self):

        self.train_path = '/home/william.hancock/workspace/data/lsdsem/ukp/train__rocstories_kde_with_features.csv'
        self.dev_path = '/home/william.hancock/workspace/data/lsdsem/ukp/dev__storycloze_valid_with_features.csv'
        self.test_path = '/home/william.hancock/workspace/data/lsdsem/ukp/test__storycloze_test_with_features.csv'

        self.load_data()


    def load_data(self):

        train_data = self.parse_csv(self.train_path)
        dev_data = self.parse_csv(self.dev_path)
        

        # take a subset of train
        train_pivot = int(len(train_data) * .05)
        self.train_data = train_data[:train_pivot]

        dev_pivot = int(len(dev_data) * .1)
        self.dev_data = dev_data[:dev_pivot]

        # build a tensor that maps words in our data to indices
        self.corpus = self.train_data + self.dev_data
        self.corpus_freq = self.build_freq_dict()




    def count(self):
        """ return size of corpus """
        return len(self.corpus_freq.keys())


    def get_good_bad_split(self, embedding, story_list):
        """
        take each story and generate two training examples from it:

        story_body      ending_one      ending_one_features     ending_one_label
        story_body      ending_two      ending_two_features     ending_two_label
        ...
        """

        examples = []

        for story in story_list:

            context_embedded = embedding.embed(story.get_context_tokens(), 80)

            for (idx, ending_tokens) in enumerate(story.get_tokenized_endings()):

                ending_embedded = embedding.embed(ending_tokens, 20)
                label = [1,0] if story.ending_idx==idx else [0,1]

                examples.append((context_embedded, ending_embedded, [None], label))


        return examples



    def get_dev_repr(self, embedding, story_list):

        """
        generate one example from each story

        story_body  ending_one  ending_one_feats    ending_two  ending_two_feats    correct_label
        ...

        """

        examples = []

        for story in story_list:

            ending_one = story.get_tokenized_endings()[0]
            ending_two = story.get_tokenized_endings()[1]

            context_embedded = embedding.embed(story.get_context_tokens(), 80)
            ending_one_embedded = embedding.embed(ending_one, 20)
            ending_two_embedded = embedding.embed(ending_two, 20)

            label = [1,0] if story.ending_idx==0 else [0,1]

            examples.append((context_embedded, ending_one_embedded, [None], ending_two_embedded, [None], label))

        return examples





    def get_vocab(self):
        return self.corpus_freq.keys()


    def build_freq_dict(self):
        """ 
            purely a cosmetic thing: build a token counter to assign IDs to
            tokens in an ordinal manner
        """

        tokens = Counter()

        for story in self.corpus:
            tokens.update(story.get_freq())

        return tokens




    def parse_csv(self, path):
        """ parse the csv files """

        stories = []

        with open(path, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = next(reader)
            num_elems = len(header)

            for row in list(reader):

                story_id = row[0]
                sentences = []
                for text in row[1:5]:
                    sentences.append(text)
                potential_endings = []
                for text in row[5:7]:
                    potential_endings.append(text)

                correct_ending_idx = int(row[7]) - 1

                stories.append(Story(story_id, sentences, potential_endings, correct_ending_idx))


        return stories
