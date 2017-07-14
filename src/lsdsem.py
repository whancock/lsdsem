import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer

class Moo:

    def __init__(self):


        self.lstm_cell_size = 141
        self.sentence_length = 20
        self.trainable_embeddings = False

        self.n_features = data.n_features




        # self.build_input(data, sess)
        self.input_story_begin = tf.placeholder(tf.int32, [None, self.sentence_length * 4])
        self.input_story_end = tf.placeholder(tf.int32, [None, self.sentence_length])
        self.input_features = tf.placeholder(tf.float32, [None, data.n_features])
        self.input_label = tf.placeholder(tf.int32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(data.embeddings)
            embeddings_weight = tf.get_variable("embeddings", data.embeddings.shape, dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.trainable_embeddings)

            self.embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            self.embeddings_story_end = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end)



        # self.initialize_weights()
        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        # self.lstm_cell_size * 2 * 2:
        # the first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        #  The second "2" is because we feed the story context and an ending * separately *, thus
        # obtaining two outputs from the LSTM.

        
        self.dense_1_W = tf.get_variable('dense_1_W', shape=[self.lstm_cell_size * 2 * 2 + self.n_features, self.lstm_cell_size], initializer=xavier_initializer())
        self.dense_1_b = tf.get_variable('dense_1_b', shape=[self.lstm_cell_size], initializer=tf.constant_initializer(.1))

        self.dense_2_W = tf.get_variable('dense_2_W', shape=[self.lstm_cell_size, 2], initializer=xavier_initializer())
        self.dense_2_b = tf.get_variable('dense_2_b', shape=[2], initializer=tf.constant_initializer(.1))



        beginning_lstm = tf.nn.dropout(
            self.apply_lstm(
                self.embeddings_story_begin,
                self.input_story_begin,
                re_use_lstm=False
            ),
            self.dropout_keep_prob
        )

        ending_lstm = tf.nn.dropout(
            self.apply_lstm(
                self.embeddings_story_end,
                self.input_story_end,
                re_use_lstm=True
            ),
            self.dropout_keep_prob
        )

        concatenated = tf.concat([beginning_lstm, ending_lstm, self.input_features], 1)

        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, self.dense_1_W, self.dense_1_b))
        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, self.dense_2_W, self.dense_2_b)




        # create outputs function
        self.loss_individual = tf.nn.softmax_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        self.loss = tf.reduce_mean(self.loss_individual)

        self.dense_2_out = tf.nn.softmax(dense_2_out)
        tf.summary.scalar('Loss', self.loss)





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






def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))