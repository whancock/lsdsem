import numpy as np

import gensim

class WVEmbedding:


    start_idx = 1
    oov_idx = 2
    word_idx_start = 3

    vocab_to_index = {}


    def __init__(self, corpus):

        self.corpus = corpus

        self.data_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/glove.6B/w2v.6B.100d.txt'
        # self.data_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/text8/text8'
        self.read_data()
        self.setup_vocab()


    def read_data(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.data_path)
        self.embedding_dim = self.model.vector_size

        

    def setup_vocab(self):

        zero_padding = np.zeros((self.embedding_dim,))
        oov = np.random.uniform(-1.0, 1.0, [self.embedding_dim, ])

        word_to_idx = {}
        idx_to_embedding = [zero_padding, oov]

        for word in self.corpus.get_vocab():

            if word in self.model.vocab:
                word_to_idx[word] = len(idx_to_embedding)
                idx_to_embedding.append(self.model[word])
            else:
                word_to_idx[word] = 1

        self.idx_to_embedding = np.array(idx_to_embedding)
        self.word_to_idx = word_to_idx



    def vocab_size(self):
        return len(self.idx_to_embedding)


    def word_to_embedding(self, word):
        return self.model[word]


    def word_to_index(self, word):
        return self.word_to_idx[word]


    def embed(self, word_list, max_len):
        return self.pad(np.array([self.word_to_index(word) for word in word_list]), max_len)



    def pad(self, vector, max_len):
        return np.pad(vector, (0, max_len - len(vector)), 'constant')

