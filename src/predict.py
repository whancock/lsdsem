# import sys
# import logging

# import tensorflow as tf



# # setup a logger
# logger = logging.getLogger('neural_network')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# handler_stdout = logging.StreamHandler(sys.stdout)
# handler_stdout.setFormatter(formatter)

# logger.addHandler(handler_stdout)
# logger.setLevel(logging.INFO)




# logger.info('Loading data')
# data = RocDataset()
# logger.info('Loading embeddings')
# embedding = WVEmbedding(data)


# saver = tf.train.Saver()

# ckpt = tf.train.get_checkpoint_state('/home/william.hancock/workspace/lsd/checkpoints')
# saver.restore(sess, ckpt.model_checkpoint_path)




# logger.info('Creating test data')
# sentence_len = 20



# dev_examples = data.get_dev_repr(embedding, data.test_data)


# # for story in data.dataset.test.stories:
# #     beginning_vecs = data.get_item_vector(story.sentences, sentence_len * 4)
# #     ending_1_vecs = data.get_item_vector([story.potential_endings[0]], sentence_len)
# #     ending_1_features = np.array(story.potential_endings[0].metadata.get('feature_values', []))
# #     ending_2_vecs = data.get_item_vector([story.potential_endings[1]], sentence_len)
# #     ending_2_features = np.array(story.potential_endings[1].metadata.get('feature_values', []))
# #     label = story.correct_endings[0]
# #     story_id = story.id
# #     # TODO: add the story id part
# #     self.data.append(
# #         (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features, label, story_id)
# #     )

# gold = []
# pred = []

# for idx, (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features, label, story_id) in enumerate(dev_examples, start=1):

#     predict, = sess.run([model.predict], feed_dict={
#         model.input_story_begin: [beginning_vecs] * 2,
#         model.input_story_end: [ending_1_vecs, ending_2_vecs],
#         # model.input_features: [ending_1_features, ending_2_features],
#         model.dropout_keep_prob: 1.0})

#     label_prediction = 0 if predict[0][0] > predict[1][0] else 1
#     gold.append(label)
#     pred.append((story_id, label_prediction))



# num_stories = len(gold)
# assert num_stories == len(pred)
# output_fn = os.path.join(self.config_global['checkpoint_dir'], 'answer.txt')
# correct = 0
# header = ["InputStoryid","AnswerRightEnding"]




# with open(output_fn, "w") as out:
#     writer = csv.writer(out, delimiter=',', quotechar='"')
#     writer.writerow(header)
#     for idx, prediction in enumerate(pred):
#         story_id, pred_label = prediction
#         writer.writerow([story_id] + [pred_label+1])

#         if pred_label == gold[idx]:
#             correct +=1

# logger.info("Done. Result: %0.2f" % (correct/num_stories * 100))