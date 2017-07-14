import keras
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import sequence

from dataset import RocDataset
from embedding import WVEmbedding


embedding_dim = 100
story_max_len = 100



dataset = RocDataset()
embedding = WVEmbedding()



# pull data out of our model in the correct format
(train_contexts, train_endings, train_labels) = dataset.get_good_bad_split(dataset.train_data)
(test_contexts, test_endings, test_labels) = dataset.get_good_bad_split(dataset.dev_data)

train_context_embeddings = embedding.get_data_embedded(train_contexts)
train_ending_embeddings = embedding.get_data_embedded(train_endings)

test_context_embeddings = embedding.get_data_embedded(test_contexts)
test_ending_embeddings = embedding.get_data_embedded(test_endings)


# pad data so that dimensions are uniform
train_context_embeddings_padded = sequence.pad_sequences(train_context_embeddings, maxlen=story_max_len)
train_ending_embeddings_padded = sequence.pad_sequences(train_ending_embeddings, maxlen=story_max_len)


# print('shape of train context:', train_context_embeddings_padded.shape)
# print('shape of train endings:', train_ending_embeddings_padded.shape)



context = Input(shape=(story_max_len, embedding_dim))
ending = Input(shape=(story_max_len, embedding_dim))


context_input = Input(shape=(100,), dtype='int32')
ending_input = Input(shape=(100,), dtype='int32')


embedding_layer = embedding.get_keras_layer()


encoded_context = embedding_layer(context_input)
encoded_ending = embedding_layer(ending_input)


shared_lstm = Bidirectional(LSTM(141))



# # When we reuse the same layer instance
# # multiple times, the weights of the layer
# # are also being reused
# # (it is effectively *the same* layer)
lstm_context = shared_lstm(encoded_context)
lstm_ending = shared_lstm(encoded_ending)



# # We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([lstm_context, lstm_ending], axis=-1)



# # And add a logistic regression on top
h_layer = Dense(141, activation='relu')(merged_vector)
output_layer = Dense(1, activation='softmax')(h_layer)

# # We define a trainable model linking the
# # tweet inputs to the predictions
model = Model(inputs=[context_input, ending_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit([train_context_embeddings_padded, train_ending_embeddings_padded], train_labels, epochs=10)