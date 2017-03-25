import tensorflow as tf
import numpy

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

HIDDEN_NODES = 20

x_ = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_uniform([784, HIDDEN_NODES], -1, 1))
b1 = tf.Variable(tf.zeros([HIDDEN_NODES]))
h1 = tf.nn.relu(tf.matmul(x_, W1) + b1)

W2 = tf.Variable(tf.random_uniform([HIDDEN_NODES, 10], -1, 1))
b2 = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(h1, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training, doesn't really converge. Randomly outputs value 0-9
for i in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Use normal batch for an iteration, then an inverted batch for an iteration
    if i% 10 < 6:
        batch_xs = numpy.subtract(1, batch_xs)
    # Print what's happening every thousand images
    if i % 1000 == 0:
        print i
        print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels}))
        print(sess.run(accuracy, feed_dict={x_: numpy.subtract(1, mnist.test.images), y_: mnist.test.labels}))
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})



# Print scores
print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x_: numpy.subtract(1, mnist.test.images), y_: mnist.test.labels}))
