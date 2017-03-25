import tensorflow as tf
import numpy

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_ = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x_, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Simple gradient decent optimizer doesn't work as well for negative images
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Invert batch_xs, since it's grayscale we can subtract all the values from 1 to invert them
    batch_xs = numpy.subtract(1, batch_xs)
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

negative_test_images = numpy.subtract(1, mnist.test.images)
print(sess.run(accuracy, feed_dict={x_: negative_test_images, y_: mnist.test.labels}))
