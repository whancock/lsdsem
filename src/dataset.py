
import csv
import itertools
import tensorflow as tf
import numpy as np
from story import Story
from collections import Counter

class RocDataset:

    def __init__(self):

        self.embedding_size = 100

        # self.spring_path = '/home/william.hancock/workspace/data/lsdsem/raw/rocstories_spring_2016.csv'
        # self.winter_path = '/home/william.hancock/workspace/data/lsdsem/raw/rocstories_winter_2017.csv'
        # self.test_path = '/home/william.hancock/workspace/data/lsdsem/raw/cloze_test_test_spring_2016.csv'
        # self.val_path = '/home/william.hancock/workspace/data/lsdsem/raw/cloze_test_val_spring_2016.csv'

        self.train_path = '/home/william.hancock/workspace/data/lsdsem/ukp/train__rocstories_kde_with_features.csv'
        self.dev_path = '/home/william.hancock/workspace/data/lsdsem/ukp/dev__storycloze_valid_with_features.csv'
        self.test_path = '/home/william.hancock/workspace/data/lsdsem/ukp/test__storycloze_test_with_features.csv'

        self.load_data()


    def load_data(self):

        self.train_data = self.parse_csv(self.train_path)
        self.dev_data = self.parse_csv(self.dev_path)
        self.test_data = self.parse_csv(self.test_path)

        # build a tensor that maps words in our data to indices
        # self.corpus = self.train_data + self.dev_data + self.test_data
        self.corpus = self.test_data

        # read in data from CSV files
        self.extract_labels()
        # calculate frequencies of words, also serves as mapping from word to index
        self.build_freq_dict()
        # build a vector of the corpus for feeding into a NN
        feats = self.extract_feats()
        # transform words into ints
        self.embeddings = self.embed_feats(self.token_to_id, feats)



    def extract_feats(self):
        """
        the story object has more info than we need. go from story to a vector of
        required input features for our model
        """
        return [story.get_tokens_raw() for story in self.corpus]



    def embed_feats(self, mapping, feats):
        """ go from words to indices based on mapping """

        start_char = 1
        index_from = 3

        return np.array([ [[start_char]] + [[mapping[w] + index_from for w in sent] for sent in story] for story in feats])




    def shuffle_corpus(self):

        # makes the generated bad examples n times the good ones
        bad_ratio = 4
        # what percentage of the data do we keep for training?
        train_ratio = .9

        bodies = self.embeddings[:,:5]
        endings = np.array(self.embeddings[:,5])[:, np.newaxis]


        good_stories = np.concatenate((bodies, endings), axis=1)
        good_stories_labels = np.ones_like(endings)
        

        # for n times the size of the corpus, pick a story, and then give it a bad ending
        # this makes the assumption that we want evenly distributes bad endings
        # each story is represented bad_ratio times and each endings is represented bad_ratio
        # times

        bad_bodies = np.tile(bodies,(bad_ratio,1))
        bad_endings = np.tile(endings,(bad_ratio,1))

        np.random.shuffle(bad_endings)

        bad_stories = np.concatenate((bad_bodies, bad_endings), axis=1)
        bad_stories_labels = np.zeros_like(bad_endings)


        all_stories = np.vstack((good_stories, bad_stories))
        all_stories_labels = np.vstack((good_stories_labels, bad_stories_labels))
        
        """
        if bad ratio is 4, then we now have

        [good bodies] [good endings]
        [bad bodies] [bad endings]
        [bad bodies] [bad endings]
        [bad bodies] [bad endings]
        [bad bodies] [bad endings]

        """



        # compress all stories into one column (I.E. we remove sentence demarcations)
        all_stories = np.array([list(itertools.chain.from_iterable(story)) for story in all_stories])



        # now shuffle them
        seed = 113
        np.random.seed(seed)
        np.random.shuffle(all_stories)
        np.random.seed(seed)
        np.random.shuffle(all_stories_labels)


        # split into train and test sets
        pivot = int(train_ratio * all_stories.shape[0])

        train_x, test_x = np.split(all_stories, [pivot])
        train_labels, test_labels = np.split(all_stories_labels, [pivot])

        return (train_x, train_labels, test_x, test_labels)






    def extract_labels(self):
        self.labels = [story.ending_idx for story in self.corpus]


    
    def build_freq_dict(self):

        tokens = Counter()

        for story in self.corpus:
            tokens.update(story.get_tokens())

        self.corpus_freq = tokens


        word_to_id = {}

        for idx, token in enumerate(self.corpus_freq.most_common()):
            word_to_id[token[0]] = idx

        self.token_to_id = word_to_id






    def parse_csv(self, path):

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



# we need to create a dictionary


moo = RocDataset()