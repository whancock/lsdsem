import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer

from util import non_zero_tokens

class UKPModel:

    LSTM_CELL_SIZE = 141
    SENTENCE_LENGTH = 20
    EPOCH_LEARNING_RATE = .0001
    DROPOUT_KEEP_PROB_VAL = .7
    N_FEATURES = 1 # TODO: make this dynamic
    TRAINABLE_EMBEDDINGS = False

    def __init__(self, data, embedding):

        # self.data = data
        self.embedding = embedding
        self.setup()


    def setup(self):

        self.input_story_begin = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH * 4], name='context')
        self.input_story_end = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH], name='end')
        # self.input_features = tf.placeholder(tf.float32, [None, self.N_FEATURES], name='feats')
        self.input_label = tf.placeholder(tf.int32, [None, 2], name='label')
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])



        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)




        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(self.embedding.idx_to_embedding)
            embeddings_weight = tf.get_variable("embeddings", 
                                                self.embedding.idx_to_embedding.shape, 
                                                dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.TRAINABLE_EMBEDDINGS)


            embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            embeddings_story_end = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end)



        beginning_lstm = tf.nn.dropout(
            self.apply_lstm(
                embeddings_story_begin,
                self.input_story_begin,
                re_use_lstm=False
            ),
            self.dropout_keep_prob
        )

        ending_lstm = tf.nn.dropout(
            self.apply_lstm(
                embeddings_story_end,
                self.input_story_end,
                re_use_lstm=True
            ),
            self.dropout_keep_prob
        )

        concatenated = tf.concat([beginning_lstm, ending_lstm], 1)
        # concatenated = tf.concat([beginning_lstm, ending_lstm, self.input_features], 1)



        

        # self.lstm_cell_size * 2 * 2:
        # the first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        #  The second "2" is because we feed the story context and an ending * separately *, thus
        # obtaining two outputs from the LSTM.


        self.dense_1_W = tf.get_variable('dense_1_W', shape=[self.LSTM_CELL_SIZE * 2 * 2, self.LSTM_CELL_SIZE], initializer=xavier_initializer())
        # self.dense_1_W = tf.get_variable('dense_1_W', shape=[self.LSTM_CELL_SIZE * 2 * 2 + self.N_FEATURES, self.LSTM_CELL_SIZE], initializer=xavier_initializer())
        
        
        self.dense_1_b = tf.get_variable('dense_1_b', shape=[self.LSTM_CELL_SIZE], initializer=tf.constant_initializer(.1))

        self.dense_2_W = tf.get_variable('dense_2_W', shape=[self.LSTM_CELL_SIZE, 2], initializer=xavier_initializer())
        self.dense_2_b = tf.get_variable('dense_2_b', shape=[2], initializer=tf.constant_initializer(.1))



        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, self.dense_1_W, self.dense_1_b))

        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, self.dense_2_W, self.dense_2_b)


       
        self.loss_individual = tf.nn.softmax_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        self.loss = tf.reduce_mean(self.loss_individual)

        self.dev_loss = tf.nn.softmax(dense_2_out)
        tf.summary.scalar('Loss', self.loss)


        optimizer = tf.train.AdamOptimizer(self.EPOCH_LEARNING_RATE)
        self.train = optimizer.minimize(self.loss)



    def train_batch(self, sess, examples, summary):

        # TODO: this data needs to be split into two training examples
        context, end_one, end_one_feats, label = zip(*examples)

        # feats = np.concatenate((np.array(end_one_feats), np.array(end_two_feats)), axis=1)

        return sess.run(
            [self.train, self.loss, self.loss_individual, summary],
            feed_dict={
                self.learning_rate: self.EPOCH_LEARNING_RATE,
                self.input_story_begin: context,
                self.input_story_end: end_one,
                self.input_label: label,
                # self.input_features: end_one_feats,
                self.dropout_keep_prob: self.DROPOUT_KEEP_PROB_VAL
            })




    def predict(self, sess, example):

        context, end_one, end_one_feats, end_two, end_two_feats, shared_feats, label = example

        return sess.run([self.dev_loss], feed_dict = {
            self.input_story_begin: [context] * 2,
            self.input_story_end: [end_one, end_two],
            # self.input_features: feats,
            self.dropout_keep_prob: 1.0
        })







    def apply_lstm(self, item, indices, re_use_lstm):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :param re_use_lstm: should be False for the first call, True for al subsequent ones to get the same lstm variables
        :return: representation tensor
        """
        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope('lstm', reuse=re_use_lstm):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                item,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

        return tf.concat([last_state[0][0], last_state[1][0]], 1)

