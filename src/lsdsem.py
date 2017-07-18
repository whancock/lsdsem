import math
import progressbar
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer
from keras.preprocessing import sequence

from dataset import RocDataset
from embedding import WVEmbedding


class Moo:

    def __init__(self):


        self.lstm_cell_size = 141
        self.sentence_length = 20
        self.trainable_embeddings = False


        data = RocDataset()
        embedding = WVEmbedding()


        (train_contexts, train_endings, train_labels) = data.get_good_bad_split(data.train_data)

        train_context_embeddings = embedding.get_data_embedded(train_contexts)
        train_ending_embeddings = embedding.get_data_embedded(train_endings)

        train_context_embeddings_padded = sequence.pad_sequences(train_context_embeddings, maxlen=80)
        train_ending_embeddings_padded = sequence.pad_sequences(train_ending_embeddings, maxlen=20)


        self.examples = list(zip(train_context_embeddings_padded, train_ending_embeddings_padded, train_labels))
        # self.n_features = data.n_features()


        # self.build_input(data, sess)
        self.input_story_begin = tf.placeholder(tf.int32, [None, self.sentence_length * 4])
        self.input_story_end = tf.placeholder(tf.int32, [None, self.sentence_length])
        # self.input_features = tf.placeholder(tf.float32, [None, data.n_features()])
        self.input_label = tf.placeholder(tf.int32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(embedding.model.syn0)
            embeddings_weight = tf.get_variable("embeddings", 
                                                embedding.model.syn0.shape, 
                                                dtype=tf.float32,
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

        
        self.dense_1_W = tf.get_variable('dense_1_W', shape=[self.lstm_cell_size * 2 * 2, self.lstm_cell_size], initializer=xavier_initializer())
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

        concatenated = tf.concat([beginning_lstm, ending_lstm], 1)

        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, self.dense_1_W, self.dense_1_b))
        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, self.dense_2_W, self.dense_2_b)




        # create outputs function
        model_loss_individual = tf.nn.softmax_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        model_loss = tf.reduce_mean(model_loss_individual)

        model_dense_2_out = tf.nn.softmax(dense_2_out)
        tf.summary.scalar('Loss', model_loss)
        self.summary = tf.contrib.deprecated.merge_all_summaries(key='summaries')

        # aka model_dense_2_out
        # model_predict = tf.nn.softmax(dense_2_out)





        # prepare next epoch function
        self.batch_i = 0


        # START FUNCTION
        self.n_epochs = 30
        self.batchsize = 40
        self.epoch_learning_rate = .01
        self.global_step = 0
        dropout_keep_prob_val = .7

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train = optimizer.minimize(model_loss)
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            
            for epoch in range(1, self.n_epochs + 1):

                for _ in range(self.get_n_batches(data)):
                    self.global_step += self.batchsize
                    train_story_begin, train_story_end, train_label = self.get_next_batch(data)

                    _, loss, loss_individual, summary = sess.run(
                        [train, model_loss, model_loss_individual, self.summary],
                        feed_dict={
                            learning_rate: self.epoch_learning_rate,
                            self.input_story_begin: train_story_begin,
                            self.input_story_end: train_story_end,
                            self.input_label: train_label,
                            self.dropout_keep_prob: dropout_keep_prob_val
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




            

    def get_n_batches(self, data):
        return int(math.ceil(data.count() / self.batchsize))

    def get_next_batch(self, data):
        """We just return the next batch data here

        :return: story beginning, story end, label
        :rtype: list, list, list
        """

        epoch_random_indices = np.random.permutation(data.count())

        indices = epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]

        data = [self.examples[i] for i in indices]





        batch_story_begin, batch_story_end, batch_label = zip(*data)
        self.batch_i += 1


        print("LEN CHECK ", len(batch_story_begin), len(batch_story_end))

        return batch_story_begin, batch_story_end, batch_label



def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))



def _create_progress_bar(dynamic_msg=None):
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]
    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))
    return progressbar.ProgressBar(widgets=widgets)




Moo()