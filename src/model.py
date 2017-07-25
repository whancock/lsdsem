import os
import math
import numpy as np
import tensorflow as tf

from util import _create_progress_bar, non_zero_tokens

from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import xavier_initializer


class LSDModel:

    __summary = None
    checkpoint_dir = '/home/william.hancock/workspace/lsd/checkpoints'

    lstm_cell_size = 141
    sentence_length = 20
    trainable_embeddings = False
    n_epochs = 30
    batchsize = 40
    epoch_learning_rate = .0001
    dropout_keep_prob_val = .7


    def __init__(self, data, embedding):

        # self.data = data
        self.embedding = embedding
        self.setup()


    def setup(self):

        self.input_story_begin = tf.placeholder(tf.int32, [None, self.sentence_length * 4])
        self.input_story_end = tf.placeholder(tf.int32, [None, self.sentence_length])
        # self.input_features = tf.placeholder(tf.float32, [None, data.n_features()])
        self.input_label = tf.placeholder(tf.float32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)


        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(self.embedding.idx_to_embedding)
            embeddings_weight = tf.get_variable("embeddings", 
                                                self.embedding.idx_to_embedding.shape, 
                                                dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.trainable_embeddings)

            self.embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            self.embeddings_story_end = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end)




        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)


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

        # self.lstm_cell_size * 2 * 2:
        # the first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        #  The second "2" is because we feed the story context and an ending * separately *, thus
        # obtaining two outputs from the LSTM.
        dense_1_W = tf.get_variable('dense_1_W', shape=[self.lstm_cell_size * 2 * 2, self.lstm_cell_size], initializer=xavier_initializer())
        dense_1_b = tf.get_variable('dense_1_b', shape=[self.lstm_cell_size], initializer=tf.constant_initializer(.1))

        dense_2_W = tf.get_variable('dense_2_W', shape=[self.lstm_cell_size, 2], initializer=xavier_initializer())
        dense_2_b = tf.get_variable('dense_2_b', shape=[2], initializer=tf.constant_initializer(.1))

        # layer H
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, dense_1_W, dense_1_b))
        # layer O
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, dense_2_W, dense_2_b)


        # create outputs function
        # model_loss_individual = tf.nn.softmax_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        self.loss_individual = tf.nn.sigmoid_cross_entropy_with_logits(logits=dense_2_out, labels=self.input_label)
        self.train_loss = tf.reduce_mean(self.loss_individual)

        # this is for validation
        self.dev_loss = tf.nn.softmax(dense_2_out)

        tf.summary.scalar('Loss', self.train_loss)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.train_loss)


        






    def train_model(self, logger, train_examples, dev_examples):

        best_val_score = 0.0
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
        
            for epoch in range(1, self.n_epochs + 1):

                logger.info('Epoch {}'.format(epoch))

                bar = _create_progress_bar('loss')
                train_losses = []  # used to calculate the epoch train loss
                recent_train_losses = []  # used to calculate the display loss

                # only thing useful from prepare_next_epoch()
                self.batch_i = 0
                self.epoch_random_indices = np.random.permutation(len(train_examples))

                for _ in bar(range(self.get_n_batches(train_examples))):

                    train_story_begin, train_story_end, train_story_end_feats, train_label = self.get_next_batch(train_examples)

                    _, loss, loss_individual, summary = sess.run(
                        [self.train, self.train_loss, self.loss_individual, self.summary],
                        feed_dict={
                            self.learning_rate: self.epoch_learning_rate,
                            self.input_story_begin: train_story_begin,
                            self.input_story_end: train_story_end,
                            self.input_label: train_label,
                            self.dropout_keep_prob: self.dropout_keep_prob_val
                        })

                    recent_train_losses = ([loss] + recent_train_losses)[:20]
                    train_losses.append(loss)
                    bar.dynamic_messages['loss'] = np.mean(recent_train_losses)


                valid_score = self.evaluate(sess, dev_examples)
                logger.info('valid score: {}'.format(valid_score))

                if valid_score > best_val_score:

                    score_val = np.around(valid_score, decimals=3)
                    model_name = "%s/%0.3f.model.ckpt" % (self.checkpoint_dir, score_val)
                    saver.save(sess, model_name)

                    best_val_score = valid_score




    def test_model(self, logger, test_examples):

        saver = tf.train.Saver()

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            print('Checkpoint dir is', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Run the model to get predictions
            logger.info('Analyzing test data')
            test_score = self.evaluate(sess, test_examples)
            logger.info("Done. Result: %0.2f" % (test_score * 100.))




    def evaluate(self, sess, examples):

        correct = 0
        total = len(examples)

        for idx, (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features,label) in enumerate(examples, start=1):

            predict, = sess.run([self.dev_loss], feed_dict = {
                self.input_story_begin: [beginning_vecs] * 2,
                self.input_story_end: [ending_1_vecs, ending_2_vecs],
                # model.input_features: [ending_1_features, ending_2_features],
                self.dropout_keep_prob: 1.0
            })

            prediction = 0 if predict[0][0] > predict[1][0] else 1

            if prediction == label:
                correct += 1


        val_score = correct / float(total)

        return val_score





    @property
    def summary(self):
        if self.__summary is None:
            self.__summary = tf.summary.merge_all(key='summaries')
        return self.__summary



    def get_n_batches(self, examples):
        return int(math.ceil(len(examples) / self.batchsize))




    def get_next_batch(self, examples):
        """We just return the next batch data here

        :return: story beginning, story end, label
        :rtype: list, list, list
        """
        indices = self.epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]
        data = [examples[i] for i in indices]
        batch_story_begin, batch_story_end, batch_story_end_feats, batch_label = zip(*data)
        self.batch_i += 1
        return batch_story_begin, batch_story_end, batch_story_end_feats, batch_label




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