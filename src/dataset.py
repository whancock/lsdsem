
import csv
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

        self.extract_labels()
        self.build_freq_dict()

        self.shuffle_corpus()
        self.build_embeddings()



    def shuffle_corpus(self):

        # makes the generated bad examples n times the good ones
        bad_ratio = 4
        # what percentage of the data do we keep for training?
        train_ratio = .9

        bodies = []
        endings = []
        for story in self.corpus:
            bodies.append(story.sentences)
            endings.append(story.endings[story.ending_idx])

        bodies = np.array(bodies)
        endings = np.array(endings)[:, np.newaxis]


        good_stories = np.concatenate((bodies, endings), axis=1)
        good_stories_labels = np.ones_like(endings)
        

        # for n times the size of the corpus, pick a story, and then give it a bad ending
        # for idx in range(len(shuffle) * bad_ratio):

        bad_bodies = np.tile(bodies,(bad_ratio,1))
        bad_endings = np.tile(endings,(bad_ratio,1))

        np.random.shuffle(bad_endings)

        bad_stories = np.concatenate((bad_bodies, bad_endings), axis=1)
        bad_stories_labels = np.zeros_like(bad_endings)

        # print(bad_stories.shape)

        all_stories = np.vstack((good_stories, bad_stories))
        all_stories_labels = np.vstack((good_stories_labels, bad_stories_labels))
        
        seed = 113
        np.random.seed(seed)
        np.random.shuffle(all_stories)
        np.random.seed(seed)
        np.random.shuffle(all_stories_labels)

        # return train and test sets
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



    def build_embeddings(self):

        start_char = 1
        index_from = 3

        self.numerized = [[start_char] + [self.token_to_id[w] + index_from for w in story.get_tokens()] for story in self.corpus]



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