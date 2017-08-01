import gensim
import numpy as np

from collections import OrderedDict


class WVEmbedding:


    start_idx = 1
    oov_idx = 2
    word_idx_start = 3

    def __init__(self, corpus):

        self.corpus = corpus

        self.data_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/glove.6B/glove.6B.100d.txt'
        # self.data_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/text8/text8'
        self.read_data()
        self.setup_vocab()


    def read_data(self):

        self.embedding_dim = 100

        zero_padding = np.zeros((self.embedding_dim,))
        oov = np.random.uniform(-1.0, 1.0, [self.embedding_dim, ])
        idx_to_embedding = [zero_padding, oov]

        word_to_idx = {}
        word_to_embedding = {}


        with open(self.data_path, 'r') as f:
            for line in f:
                if line:

                    try:

                        tv = line.strip().split(" ", 1)

                        assert len(tv) == 2

                        word = tv[0]
                        vector = np.fromstring(tv[1], sep=' ')

                        assert len(vector) == self.embedding_dim

                        word_to_idx[tv[0]] = len(idx_to_embedding)
                        idx_to_embedding.append(vector)
                        word_to_embedding[word] = vector

                    except ValueError:
                        print(tv)


        self.idx_to_embedding = np.array(idx_to_embedding)
        self.word_to_embedding = word_to_embedding
        self.word_to_idx = word_to_idx

        

    def setup_vocab(self):
        """
        there may be words in our corpus that we didn't see in the word2vec model
        so we need to map them to the OOV vector
        """

        for word in self.corpus.get_vocab():
            if not word in self.word_to_idx:
                self.word_to_idx[word] = 1




    def vocab_size(self):
        return len(self.idx_to_embedding)

    def word_embedding(self, word):
        return self.word_to_embedding[word]

    def word_index(self, word):
        return self.word_to_idx[word]

    def embed(self, word_list, max_len):
        return self.pad(np.array([self.word_index(word) for word in word_list]), max_len)

    def pad(self, vector, max_len):
        return np.pad(vector, (0, max_len - len(vector)), 'constant')

