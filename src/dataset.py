from collections import Counter

import csv
import itertools
import numpy as np
from story import Story


class RocDataset:

    """ a class to help us use ROCStory data for machine learning """

    def __init__(self):

        self.embedding_size = 100

        self.train_path = '/home/william.hancock/workspace/data/lsdsem/ukp/train__rocstories_kde_with_features.csv'
        self.dev_path = '/home/william.hancock/workspace/data/lsdsem/ukp/dev__storycloze_valid_with_features.csv'
        self.test_path = '/home/william.hancock/workspace/data/lsdsem/ukp/test__storycloze_test_with_features.csv'

        self.load_data()


    def load_data(self):

        self.train_data = self.parse_csv(self.train_path)
        self.dev_data = self.parse_csv(self.dev_path)
        self.test_data = self.parse_csv(self.test_path)

        # build a tensor that maps words in our data to indices
        self.corpus = self.train_data + self.dev_data + self.test_data

        self.build_freq_dict()

        # read in data from CSV files
        # self.train_labels = self.extract_labels(self.train_data)
        # self.dev_labels = self.extract_labels(self.dev_data)
        # self.test_labels = self.extract_labels(self.test_data)

        # # build a vector of the corpus for feeding into a NN
        # self.train_feats = self.build_word_matrix(self.train_data)
        # self.dev_feats = self.build_word_matrix(self.dev_data)
        # self.test_feats = self.build_word_matrix(self.test_data)



    # def extract_labels(self, story_set):
    #     return [story.ending_idx for story in self.corpus]




    # def build_word_matrix(self, story_list):
    #     """
    #     the story object has more info than we need. go from story to a vector of
    #     required input features for our model
    #     """
    #     return [story.get_tokens() for story in story_list]


    def get_good_bad_split(self, story_list):
        """
        split each story into its good and bad version, keeping the context and ending
        seperate. this matches a specific NN architecture
        """
        context = []
        ending = []
        label = []

        for story in story_list:

            context_tokens = story.get_context_tokens()

            for (idx, ending_tokens) in enumerate(story.get_tokenized_endings()):
                context.append(context_tokens)
                ending.append(ending_tokens)
                label.append(int(idx==story.ending_idx))

        return (np.array(context), np.array(ending), np.array(label))
        



    def get_data_embedded(self, data, embedding):
        return [embedding.embed(sent) for sent in data]



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






    # def report_props(self):

    #     maxlen = 0
    #     for story in self.embeddings:
    #         curlen = sum([len(sent) for sent in story])
    #         if curlen > maxlen:
    #             maxlen = curlen

    #     vocab_size = len(self.token_to_id)

    #     print("longest story:", maxlen)
    #     print("vocab size:", vocab_size)




    # def embed_feats(self, embedding, feats):
    #     """ go from words to indices based on mapping """
    #     return np.array([ [[mapping[w] + index_from for w in sent] for sent in story] for story in feats])



    # def get_data(self):
    #     """ process corpus and return test and train sets """

    #     (train_base, test_base) = self.split_test_train()

    #     (train_x, train_y) = self.more_neg_examples(train_base)
    #     (test_x, test_y) = self.more_neg_examples(test_base, bad_ratio=2)

    #     return (train_x, train_y, test_x, test_y)




    # def split_test_train(self):

    #     # what percentage of the data do we keep for training?
    #     train_ratio = .9

    #     # split into train and test sets
    #     pivot = int(train_ratio * self.embeddings.shape[0])

    #     return np.split(self.embeddings, [pivot])



    # def more_neg_examples(self, stories, bad_ratio=4):

    #     """
    #     take as input all positive training examples (because that's what our corpus looks like)
    #     and use this as a base to generate noise for better model fitting. bad_ratio is the ratio
    #     of noise to signal
    #     """

    #     bodies = stories[:,:5]
    #     endings = np.array(stories[:,5])[:, np.newaxis]


    #     good_stories = np.concatenate((bodies, endings), axis=1)
    #     good_stories_labels = np.ones_like(endings)
        

    #     # for n times the size of the corpus, pick a story, and then give it a bad ending
    #     # this makes the assumption that we want evenly distributes bad endings
    #     # each story is represented bad_ratio times and each endings is represented bad_ratio
    #     # times

    #     bad_bodies = np.tile(bodies,(bad_ratio,1))
    #     bad_endings = np.tile(endings,(bad_ratio,1))

    #     np.random.shuffle(bad_endings)

    #     bad_stories = np.concatenate((bad_bodies, bad_endings), axis=1)
    #     bad_stories_labels = np.zeros_like(bad_endings)


    #     all_stories = np.vstack((good_stories, bad_stories))
    #     all_stories_labels = np.vstack((good_stories_labels, bad_stories_labels))
        
    #     """
    #     if bad ratio is 4, then we now have

    #     [good bodies] [good endings]
    #     [bad bodies] [bad endings]
    #     [bad bodies] [bad endings]
    #     [bad bodies] [bad endings]
    #     [bad bodies] [bad endings]

    #     """


    #     # compress all stories into one column (I.E. we remove sentence demarcations)
    #     all_stories = np.array([list(itertools.chain.from_iterable(story)) for story in all_stories])


    #     # now shuffle them
    #     seed = 113
    #     np.random.seed(seed)
    #     np.random.shuffle(all_stories)
    #     np.random.seed(seed)
    #     np.random.shuffle(all_stories_labels)

    #     return (all_stories, all_stories_labels)



    
    def build_freq_dict(self):
        """ 
            purely a cosmetic thing: build a token counter to assign IDs to
            tokens in an ordinal manner
        """

        tokens = Counter()

        for story in self.corpus:
            tokens.update(story.get_freq())

        self.corpus_freq = tokens


        word_to_id = {}
        id_to_word = {}

        for idx, token in enumerate(self.corpus_freq.most_common()):
            word_to_id[token[0]] = idx
            id_to_word[idx] = token[0]

        self.token_to_id = word_to_id
        self.id_to_token = id_to_word



    # def validate(self, feats, labels):
    #     """
    #     checks to see if our int mapping worked out and the shuffling is cool by printing
    #     out random stories from the corpus and relying on the coder to see if the label makes
    #     sense
    #      """

    #     print(feats.shape)
    #     print(labels.shape)

    #     for idx in range(20):

    #         feat = feats[idx]
    #         label = labels[idx]

    #         story = []

    #         for word_idx in feat:
    #             if word_idx >= 3:
    #                 story.append(self.id_to_token[word_idx - 3])

    #         print(' '.join(story))
    #         print(label)
    #         print('\n')

