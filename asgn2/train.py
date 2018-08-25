# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from nn import NN

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 512, 'the batch_size of the training procedure')
flags.DEFINE_integer('epochs', 1500, 'epochs')
flags.DEFINE_float('learning_rate', 0.0015, 'the learning rate')
flags.DEFINE_integer('checkpoint_num', 1000, 'epoch num of checkpoint')
flags.DEFINE_integer('input_size', 784, 'input shape')
flags.DEFINE_integer('hidden_size', 625, 'hidden neural size')
flags.DEFINE_integer('output_size', 10, 'output number of classes')
flags.DEFINE_float('keep_prob', 0.9, 'dropout rate')


class Config(object):
    input_size = FLAGS.input_size
    hidden_size = FLAGS.hidden_size
    output_size = FLAGS.output_size
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    keep_prob = FLAGS.keep_prob
    checkpoint_num = FLAGS.checkpoint_num
    epochs = FLAGS.epochs


def train_step():
    mnist = input_data.read_data_sets("{}/data".format(os.path.abspath(os.path.dirname(__file__))), one_hot=True)
    config = Config()
    eval_config = Config()
    eval_config.keep_prob = 1.0
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
        model = NN(config, is_training=True)
        valid_model = NN(eval_config, is_training=False)
        tf.global_variables_initializer().run()
        step = 0
        train_loss = 0
        for i in range(config.epochs):
            xs, ys = mnist.train.next_batch(config.batch_size)
            session.run(model.optimizer, feed_dict={model.x: xs, model.y_: ys})
            step += 1
            train_loss += session.run(model.loss, feed_dict={model.x: xs, model.y_: ys})
            if step % 300 == 0:
                print("After {0} training steps, loss is {1}".format(step, train_loss / step))
        test_acc = session.run(valid_model.accuracy, feed_dict={valid_model.x: mnist.test.images, valid_model.y_: mnist.test.labels})
        print("After {0} training steps, test accuracy using average model is {1}".format(eval_config.epochs, round(test_acc, 3)))

def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
