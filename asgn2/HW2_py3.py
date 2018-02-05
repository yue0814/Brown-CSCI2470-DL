#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

input_size = 784
hidden_size = 500
output_size = 10

batch_size = 1024
learning_rate = 0.0015
training_step = 5000

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])

    W1 = tf.Variable(tf.truncated_normal([input_size, hidden_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.0, shape=[hidden_size]))
    W2 = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.0, shape=[output_size]))

    A1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    A2 = tf.nn.softmax(tf.matmul(A1, W2) + b2)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=A2)
    loss = tf.reduce_mean(cross_entropy)

    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(A2, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    start = time.time()
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True    
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        step = 0
        train_loss = 0
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}  

        for i in range(training_step):
            
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: xs, y_: ys})
            step += 1
            train_loss += sess.run(loss, feed_dict={x:xs, y_: ys})
            if step % 300 == 0:
                print("After {0} training step, loss is {1}".format(step, train_loss/step))
        
        test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training steps, test accuracy using average "
           "model is %g" % (training_step, test_acc))
    end = time.time()
    print("Running time is {0}".format(end-start))
def main(argv=None): 
    
    mnist = input_data.read_data_sets("data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()