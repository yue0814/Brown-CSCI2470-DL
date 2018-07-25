#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define hyperparameters
learning_rate = 0.0001
training_iters = 2000 * 50
batch_size = 50

# Define network parameters
n_input = 784
n_classes = 10

# Placeholder
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])

# Convolution
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

# Pooling
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

weights = {
    'W1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    'W2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    'W4': tf.Variable(tf.truncated_normal([64 * 7 * 7, 784], stddev=0.1)),
    'Wo': tf.Variable(tf.truncated_normal([784, n_classes], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'b4': tf.Variable(tf.random_normal([784], stddev=0.1)),
    'bo': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}

def model(X, weights, biases):
    # Conv1
    conv1 = tf.nn.relu(conv2d('conv1', X, weights['W1'], biases['b1']))
    # Pool1
    pool1 = maxpool2d('pool1', conv1, k=2)

    # Conv2
    conv2 = tf.nn.relu(conv2d('conv2', pool1, weights['W2'], biases['b2']))

    # Pool2
    pool2 = maxpool2d('pool2', conv2, k=2)


    # Full connect layer
    fc = tf.reshape(pool2, [-1, weights['W4'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['W4']), biases['b4'])
    fc = tf.nn.relu(fc)

    # output
    a = tf.add(tf.matmul(fc, weights['Wo']), biases['bo'])

    return a


# prediction
pred = model(X, weights, biases)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step*batch_size <= training_iters:
        xs, ys = mnist.train.next_batch(batch_size)
        xs = xs.reshape(-1, 28, 28, 1)
        sess.run(optimizer, feed_dict={X:xs, y:ys})

        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:xs, y:ys})

            print("Iter {0}, Minibatch Loss = {1}, Training accuracy = {2}".format(str(step),\
                                                                                    loss, acc))
        step += 1
    print("Optimization Completed")

    print("Testing Accuracy: {0}".format(sess.run(accuracy,\
             feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), y: mnist.test.labels})))