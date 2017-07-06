import tensorflow as tf

n_inputs = 3
n_neurons = 5

# input layer for the first step in the rnn
X0 = tf.placeholder(tf.float32, [None, n_inputs])
# input layer for the second step in the rnn
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

