import tensorflow as tf
import gensim


'''
so I think we need to do something like this:::?

[ ... ... ... ... 0 0 0 ] [ 0 ]




'''



from dataset import RocDataset

embedding_dim = 100
embedding_path = '/media/william.hancock/d9b5fd30-e5fd-4df4-874e-58c039a3c6d5/models/glove.6B/w2v.6B.100d.txt'


embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
# sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
# vectors = [model[w.lower()] for w in sentence]
# print(vectors)

dataset = RocDataset()


# https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow

# W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                 trainable=False, name="W")

# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# embedding_init = W.assign(embedding_placeholder)

# sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
# tf.nn.embedding_lookup(W, input_x)