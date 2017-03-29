import tensorflow as tf
import numpy

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the first convolution layer weights
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Next layer to find 64 features (more detailed)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add dropouts, randomly drops out one of the FC layers output neurons
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final output layer, converts the 1024 inputs to 10 outputs
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train using adam optimiser
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

# Training, doesn't really converge. Randomly outputs value 0-9
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Use normal batch for an iteration, then an inverted batch for an iteration
    if i% 10 < 6:
        batch_xs = numpy.subtract(1, batch_xs)
    # Print what's happening every thousand images
    if i % 100 == 0:
        print i
        print(sess.run(accuracy, feed_dict={x_: mnist.test.images[:500], y_: mnist.test.labels[:500], keep_prob: 0.5}))
        print(sess.run(accuracy, feed_dict={x_: numpy.subtract(1, mnist.test.images[:500]), y_: mnist.test.labels[:500], keep_prob: 0.5}))
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys, keep_prob: 0.5})



# Print scores
print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5}))
print(sess.run(accuracy, feed_dict={x_: numpy.subtract(1, mnist.test.images), y_: mnist.test.labels, keep_prob: 0.5}))
