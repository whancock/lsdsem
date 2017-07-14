from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.layers.core import Activation





model = Sequential()

model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))

model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')