import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

HIDDEN_NODES = 10

x_ = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_uniform([784, HIDDEN_NODES], -1, 1))
b1 = tf.Variable(tf.zeros([HIDDEN_NODES]))
h1 = tf.nn.relu(tf.matmul(x_, W1) + b1)

W2 = tf.Variable(tf.random_uniform([HIDDEN_NODES, 10], -1, 1))
b2 = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(h1, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
for i in range(100000):
    if i%5000 == 0:
        print i
        print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels}))

    batch_xs, batch_ys = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})


print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels}))