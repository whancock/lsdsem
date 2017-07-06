import tensorflow as tf





sess_config = tf.ConfigProto(log_device_placement=False)
# sess_config.gpu_options.allow_growth = True

# we allow to set the random seed in the config file to perform multiple subsequent runs with different
# initialization values in order to compute an avg result score

with tf.Session(config=sess_config) as sess:


    learning_rate = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(model.loss)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    best_val_score = 0.0
    i = 0
    for epoch in range(1, self.n_epochs + 1):
        self.logger.info('Epoch {}/{}'.format(epoch, self.n_epochs))

        self.logger.debug('Preparing epoch')
        self.prepare_next_epoch(model, data, sess, epoch)

        bar = _create_progress_bar('loss')

    train_losses = []  # used to calculate the epoch train loss
    recent_train_losses = []  # used to calculate the display loss

    self.logger.debug('Training')
    for _ in bar(range(self.get_n_batches())):
        self.global_step += self.batchsize
        train_story_begin, train_story_end, train_features, train_label = self.get_next_batch(
            model, data, sess)

        _, loss, loss_individual, summary = sess.run(
            [train, model.loss, model.loss_individual, model.summary],
            feed_dict={
            learning_rate: self.epoch_learning_rate,
            model.input_story_begin: train_story_begin,
            model.input_story_end: train_story_end,
            model.input_label: train_label,
            model.input_features: train_features,
            model.dropout_keep_prob: self.dropout_keep_prob
            })
        recent_train_losses = ([loss] + recent_train_losses)[:20]
        train_losses.append(loss)
        bar.dynamic_messages['loss'] = np.mean(recent_train_losses)
        self.add_summary(summary)
        self.logger.info('train loss={:.6f}'.format(np.mean(train_losses)))

        self.logger.info('Now calculating validation score')
        valid_score = self.calculate_validation_score(sess, model, data)
        if valid_score > best_val_score:
            score_val = np.around(valid_score, decimals=3)
            model_name = "%s/%0.3f.model.ckpt" % (
                self.checkpoint_dir, score_val)
            self.logger.info('Saving the model into %s' % model_name)
            saver.save(sess, model_name)
            best_val_score = valid_score
            self.logger.info('Score={:.4f}'.format(valid_score))
