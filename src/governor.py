import math

import numpy as np
import tensorflow as tf

from util import _create_progress_bar


class Governor:

    """
    responsible for controlling training and testing
    """

    N_EPOCHS = 30
    BATCHSIZE = 40
    CHECKPOINT_DIR = '/home/william.hancock/workspace/lsd/checkpoints'

    def __init__(self, logger, model):

        self.logger = logger
        self.model = model

        self.saver = tf.train.Saver()



    def train_model(self, train_examples, dev_examples):

        best_val_score = 0.0

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            summary_moo = tf.summary.merge_all(key='summaries')
        
            for epoch in range(self.N_EPOCHS):

                self.logger.info('Epoch {}'.format(epoch))
                
                train_losses = []  # used to calculate the epoch train loss
                recent_train_losses = []  # used to calculate the display loss

                progress_bar = _create_progress_bar('loss')
                indices = np.random.permutation(len(train_examples))

                for batch_idx in progress_bar(range(self.get_n_batches(train_examples))):

                    batch = self.get_next_batch(train_examples, indices, batch_idx)
                    _, loss, _, _ = self.model.train_batch(sess, batch, summary_moo)

                    recent_train_losses = ([loss] + recent_train_losses)[:20]
                    train_losses.append(loss)
                    progress_bar.dynamic_messages['loss'] = np.mean(recent_train_losses)


                valid_score = self.evaluate(sess, dev_examples)
                self.logger.info('valid score: {}'.format(valid_score))

                if valid_score > best_val_score:

                    score_val = np.around(valid_score, decimals=3)
                    model_name = "%s/%0.3f.model.ckpt" % (self.CHECKPOINT_DIR, score_val)
                    self.saver.save(sess, model_name)

                    best_val_score = valid_score




    def test_model(self, test_examples):

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(self.CHECKPOINT_DIR)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

            # Run the model to get predictions
            self.logger.info('Analyzing test data')
            test_score = self.evaluate(sess, test_examples)
            self.logger.info("Done. Result: %0.2f" % (test_score * 100.))




    def evaluate(self, sess, examples):

        correct = 0
        total = len(examples)

        for example in examples:

            predict, = self.model.predict(sess, example)

            label = example[5]
            prediction = 0 if predict[0][0] > predict[0][1] else 1

            if prediction == label[1]:
                correct += 1


        val_score = correct / float(total)

        return val_score




    def get_n_batches(self, examples):
        return int(math.ceil(len(examples) / self.BATCHSIZE))



    def get_next_batch(self, examples, indices, batch_idx):
        """We just return the next batch data here

        :return: story beginning, story end, label
        :rtype: list, list, list
        """
        batch_indices = indices[batch_idx * self.BATCHSIZE: (batch_idx + 1) * self.BATCHSIZE]
        data = [examples[i] for i in batch_indices]
        context, end_one, end_one_feats, end_two, end_two_feats, label = zip(*data)

        return context, end_one, end_one_feats, end_two, end_two_feats, label