import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer

from util import non_zero_tokens

class LSDTriModel:

    LSTM_CELL_SIZE = 141
    SENTENCE_LENGTH = 20
    EPOCH_LEARNING_RATE = .0001
    DROPOUT_KEEP_PROB_VAL = .7
    N_FEATURES = 5 # TODO: make this dynamic
    TRAINABLE_EMBEDDINGS = False

    def __init__(self, data, embedding):

        # self.data = data
        self.embedding = embedding
        self.setup()


    def setup(self):

        self.input_story_begin = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH * 4])

        self.input_story_end_one = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH])
        self.input_story_end_two = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH])

        # self.input_features = tf.placeholder(tf.float32, [None, self.N_FEATURES])


        self.input_label = tf.placeholder(tf.float32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)




        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(self.embedding.idx_to_embedding)
            embeddings_weight = tf.get_variable("embeddings", 
                                                self.embedding.idx_to_embedding.shape, 
                                                dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.TRAINABLE_EMBEDDINGS)


            embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            embeddings_story_end_one = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end_one)
            embeddings_story_end_two = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end_two)



        print("embeddings_story_begin shape ", embeddings_story_begin.get_shape())
        print("input_story_begin shape ", self.input_story_begin.get_shape())


        with tf.variable_scope('begin', reuse=False):

            beginning_lstm = self.apply_lstm(
                embeddings_story_begin,
                self.input_story_begin
            )

        with tf.variable_scope('ending_one', reuse=False):
            
            ending_lstm_one = self.apply_lstm(
                embeddings_story_end_one,
                self.input_story_end_one
            )

        with tf.variable_scope('ending_two', reuse=False):

            ending_lstm_two = self.apply_lstm(
                embeddings_story_end_two,
                self.input_story_end_two
            )

        concatenated = tf.concat([beginning_lstm, ending_lstm_one, ending_lstm_two], 1)
        # concatenated = tf.concat([beginning_lstm, ending_lstm_one, ending_lstm_two, self.input_features], 1)



        # The first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        # The second "3" is because we feed the story context and two endings * separately *, thus
        # obtaining three LSTM outputs.
        dense_1_w = tf.get_variable('dense_1_W', shape=[self.LSTM_CELL_SIZE * 2 * 3, self.LSTM_CELL_SIZE], initializer=xavier_initializer())
        # dense_1_w = tf.get_variable('dense_1_W', shape=[self.LSTM_CELL_SIZE * 2 * 3  + self.N_FEATURES, self.LSTM_CELL_SIZE], initializer=xavier_initializer())
        
        dense_1_b = tf.get_variable('dense_1_b', shape=[self.LSTM_CELL_SIZE], initializer=tf.constant_initializer(.1))

        dense_2_w = tf.get_variable('dense_2_W', shape=[self.LSTM_CELL_SIZE, 2], initializer=xavier_initializer())
        dense_2_b = tf.get_variable('dense_2_b', shape=[2], initializer=tf.constant_initializer(.1))

        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, dense_1_w, dense_1_b))
        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, dense_2_w, dense_2_b)


        # create outputs function
        self.loss_individual = tf.nn.softmax_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        # self.loss_individual = tf.nn.sigmoid_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)

        
        self.train_loss = tf.reduce_mean(self.loss_individual)

        # this is for validation
        self.dev_loss = tf.nn.softmax(dense_2_out)

        tf.summary.scalar('Loss', self.train_loss)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.train_loss)




    def train_batch(self, sess, examples, summary):

        context, end_one, end_one_feats, end_two, end_two_feats, shared_feats, label = examples

        feats = np.concatenate((np.array(end_one_feats), np.array(end_two_feats), np.array(shared_feats)), axis=1)

        return sess.run(
            [self.train, self.train_loss, self.loss_individual, summary],
            feed_dict={
                self.learning_rate: self.EPOCH_LEARNING_RATE,
                self.input_story_begin: context,
                self.input_story_end_one: end_one,
                self.input_story_end_two: end_two,
                self.input_label: label,
                # self.input_features: feats,
                self.dropout_keep_prob: self.DROPOUT_KEEP_PROB_VAL
            })




    def predict(self, sess, example):

        context, end_one, end_one_feats, end_two, end_two_feats, shared_feats, label = example

        feats = np.concatenate((np.array([end_one_feats]), np.array([end_two_feats]), np.array([shared_feats])), axis=1)

        return sess.run([self.dev_loss], 
            feed_dict = {
                self.input_story_begin: [context],
                self.input_story_end_one: [end_one],
                self.input_story_end_two: [end_two],
                # self.input_features: feats,
                self.dropout_keep_prob: 1.0
            })







    def apply_lstm(self, embedding, indices):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :param reuse_lstm: should be False for the first call, True for al subsequent ones to get the same lstm variables
        :return: representation tensor
        """
        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        lstm_cell_forward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True, reuse=False)
        lstm_cell_backward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True, reuse=False)

        _, last_state = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_forward,
            lstm_cell_backward,
            embedding,
            dtype=tf.float32,
            sequence_length=sequence_length
        )

        output = tf.concat([last_state[0][0], last_state[1][0]], 1)
        return tf.nn.dropout(output, self.dropout_keep_prob)

