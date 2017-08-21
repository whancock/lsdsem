import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer

from util import non_zero_tokens

class LSDModel:

    LSTM_CELL_SIZE = 141
    SENTENCE_LENGTH = 20
    TRAINABLE_EMBEDDINGS = False
    EPOCH_LEARNING_RATE = .0001
    DROPOUT_KEEP_PROB_VAL = .7
    N_FEATURES = 5 # TODO: make this dynamic

    def __init__(self, data, embedding):

        # self.data = data
        self.embedding = embedding
        self.setup()


    def setup(self):

        self.input_story_begin = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH * 4])
        self.input_story_end = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH])
        # self.input_features = tf.placeholder(tf.float32, [None, data.n_features()])
        self.input_label = tf.placeholder(tf.float32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)




        self.input_story_begin_two = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH * 4])
        self.input_story_end_two = tf.placeholder(tf.int32, [None, self.SENTENCE_LENGTH])




        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(self.embedding.idx_to_embedding)
            embeddings_weight = tf.get_variable("embeddings", 
                                                self.embedding.idx_to_embedding.shape, 
                                                dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.TRAINABLE_EMBEDDINGS)

            embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            embeddings_story_end = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end)
            
            embeddings_story_begin_two = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin_two)
            embeddings_story_end_two = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end_two)




        # with tf.variable_scope('lstm_cell_fw'):
        self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)
        # with tf.variable_scope('lstm_cell_bw'):
        self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)


        # with tf.variable_scope('lstm_cell_fw_dos'):
        self.lstm_cell_forward_dos = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)
        # with tf.variable_scope('lstm_cell_bw_dos'):
        self.lstm_cell_backward_dos = rnn_cell.BasicLSTMCell(self.LSTM_CELL_SIZE, state_is_tuple=True)


        print("embeddings_story_begin shape ", embeddings_story_begin.get_shape())
        print("input_story_begin shape ", self.input_story_begin.get_shape())


        beginning_lstm = tf.nn.dropout(
            self.apply_lstm(
                'lstm_uno',
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                embeddings_story_begin,
                self.input_story_begin,
                reuse_lstm=None
            ),
            self.dropout_keep_prob
        )
        
        ending_lstm = tf.nn.dropout(
            self.apply_lstm(
                'lstm_uno',
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                embeddings_story_end,
                self.input_story_end,
                reuse_lstm=True
            ),
            self.dropout_keep_prob
        )

        beginning_lstm_two = tf.nn.dropout(
            self.apply_lstm(
                'lstm_dos',
                self.lstm_cell_forward_dos,
                self.lstm_cell_backward_dos,
                embeddings_story_begin_two,
                self.input_story_begin_two,
                reuse_lstm=None
            ),
            self.dropout_keep_prob
        )

        ending_lstm_two = tf.nn.dropout(
            self.apply_lstm(
                'lstm_dos',
                self.lstm_cell_forward_dos,
                self.lstm_cell_backward_dos,
                embeddings_story_end_two,
                self.input_story_end_two,
                reuse_lstm=True
            ),
            self.dropout_keep_prob
        )




        concatenated = tf.concat([beginning_lstm, ending_lstm, beginning_lstm_two, ending_lstm_two], 1)

        # self.LSTM_CELL_SIZE * 2 * 2:
        # the first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        #  The second "2" is because we feed the story context and an ending * separately *, thus
        # obtaining two outputs from the LSTM.
        dense_1_W = tf.get_variable('dense_1_W', shape=[self.LSTM_CELL_SIZE * 2 * 4, self.LSTM_CELL_SIZE], initializer=xavier_initializer())
        dense_1_b = tf.get_variable('dense_1_b', shape=[self.LSTM_CELL_SIZE], initializer=tf.constant_initializer(.1))

        dense_2_W = tf.get_variable('dense_2_W', shape=[self.LSTM_CELL_SIZE, 2], initializer=xavier_initializer())
        dense_2_b = tf.get_variable('dense_2_b', shape=[2], initializer=tf.constant_initializer(.1))

        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, dense_1_W, dense_1_b))
        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, dense_2_W, dense_2_b)


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

        context, end_one, end_one_feats, end_two, end_two_feats, shared_feats, label = zip(*examples)

        # feats = np.concatenate((np.array(end_one_feats), np.array(end_two_feats), np.array(shared_feats)), axis=1)

        return sess.run(
            [self.train, self.train_loss, self.loss_individual, summary],
            feed_dict={
                self.learning_rate: self.EPOCH_LEARNING_RATE,
                self.input_story_begin: context,
                self.input_story_begin_two: context,
                self.input_story_end: end_one,
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
                self.input_story_begin_two: [context],
                self.input_story_end: [end_one],
                self.input_story_end_two: [end_two],
                # self.input_features: feats,
                self.dropout_keep_prob: 1.0
            })








    def apply_lstm(self, scope, fw, bw, embedding, indices, reuse_lstm):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :param reuse_lstm: should be False for the first call, True for al subsequent ones to get the same lstm variables
        :return: representation tensor
        """
        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope(scope, reuse=reuse_lstm):

            # if reuse_lstm:
            #     cscope.reuse_variables()

            _, last_state = tf.nn.bidirectional_dynamic_rnn(
                fw,
                bw,
                embedding,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

        return tf.concat([last_state[0][0], last_state[1][0]], 1)