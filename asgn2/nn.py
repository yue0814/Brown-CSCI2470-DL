# -*- coding: utf-8 -*-
import tensorflow as tf


class NN(object):
    def network(self, x):
        A1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        A1 = tf.nn.dropout(A1, keep_prob=self.keep_prob)
        A2 = tf.matmul(A1, self.W2) + self.b2
        return A2

    def __init__(self, config, is_training=True):
        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.x = tf.placeholder(tf.float32, shape=[None, config.input_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, config.output_size])
        self.W1 = tf.Variable(tf.truncated_normal([config.input_size, config.hidden_size], stddev=0.1, seed=6))
        self.b1 = tf.Variable(tf.zeros([config.hidden_size]) + tf.constant(0.1))
        self.W2 = tf.Variable(tf.truncated_normal([config.hidden_size, config.output_size], stddev=0.1, seed=6))
        self.b2 = tf.Variable(tf.zeros([config.output_size]) + tf.constant(0.01))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.network(self.x)))
        # loss_summary = tf.summary.scalar("loss_summary", self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.network(self.x), 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if not is_training:
            return

        self.global_step = tf.Variable(0, trainable=False)
        # self.lr = tf.Variable(0.0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
