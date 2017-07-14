import numpy as np

import gensim

class WVEmbedding:


    start_idx = 1
    oov_idx = 2
    word_idx_start = 3


    def __init__(self):
        self.data_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/glove.6B/w2v.6B.100d.txt'
        self.read_data()


    def read_data(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.data_path)
        self.dim = self.model.vector_size

        # build the word to index lookup
        word2index = {}
        for idx, word in enumerate(self.model.index2word):
            word2index[word] = idx

        self.word2index = word2index


    def vocab_size(self):
        return len(self.model.vocab)


    def get_word_vector(self, word):
        return self.model[word]


    def word_to_index(self, word):

        if word in self.word2index:
            return self.word_idx_start + self.word2index[word]
        else:
            return self.oov_idx


    # def index_to_word(self, index):
    #     return self.model.index2word[index]


    def get_data_embedded(self, data):
        return [self.embed(sent) for sent in data]


    def embed(self, word_list):
        return np.array([self.start_idx] + [self.word_to_index(word) for word in word_list])



    def get_keras_layer(self):
        return self.model.get_embedding_layer()




# moo = Embedding('/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/glove.6B/w2v.6B.100d.txt')
# moo.read_data()
    