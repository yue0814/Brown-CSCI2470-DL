#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import gzip
import sys
# import matplotlib.pyplot as plt
# import matplotlib as mpl


def extract_images(filename, num_images):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 28, 28)
    return data


def extract_labels(filename, num_labels):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        labels = labels.reshape(1, num_labels)
    return labels

train_data_path = sys.argv[1]
train_label_path = sys.argv[2]
test_data_path = sys.argv[3]
test_label_path = sys.argv[4]

train_datas = extract_images(train_data_path, 60000)
train_labels = extract_labels(train_label_path, 60000)
test_datas = extract_images(test_data_path, 10000)
test_labels = extract_labels(test_label_path, 10000)

# show the image
# plt.imshow(train_datas[59999], cmap=mpl.cm.Greys)


# Normalize the image data
train_inputs = train_datas.reshape(train_datas.shape[0], -1).T
test_inputs = test_datas.reshape(test_datas.shape[0], -1).T

train_inputs = train_inputs / 255.
test_inputs = test_inputs / 255.

print train_inputs.shape
print train_labels.shape

# Network Structure
class Network(object):

    def __init__(self, layers_dim):
        input_dim = layers_dim[0]
        output_dim = layers_dim[1]
        self.W = np.zeros((output_dim, input_dim))
        # self.W = np.random.randn(output_dim, input_dim) * 0.001
        self.b = np.zeros((output_dim, 1))
        # self.b = np.random.randn(output_dim, 1) * 0.001

    def feed_forward(self, X):
        Z = self.W.dot(X) + self.b
        A = self.softmax(Z)
        return Z, A

    def predict(self, X):
        """Get max prob. for class"""
        Z, A = self.feed_forward(X)
        return np.argmax(A, axis=0)

    def softmax(self, Z):
        probs = np.exp(Z) / np.sum(np.exp(Z))
        return probs

    def crossentropy_loss(self, X, Y):
        Z, A = self.feed_forward(X)
        logprobs = -np.log(A[Y, range(X.shape[1])])
        return logprobs

    def back_forward(self, X, Y, learning_rate):
        Z, A = self.feed_forward(X)
        A[Y, 0] = A[Y, 0] - 1
        dW = np.dot(A, X.T)
        db = A
        self.W += -learning_rate * dW
        self.b += -learning_rate * db
        return self.W, self.b

    def train(self, X, Y, learning_rate):
        for i in range(X.shape[1]):     
            self.W, self.b = self.back_forward(X[:,i].reshape(784, 1), Y[:, i][0], learning_rate)
            if i % 1000 == 0:
                print "Loss after %s iteration is %0.8f" \
                    % (i, self.crossentropy_loss(X[:,i].reshape(784, 1), Y[:, i][0]))
        return self.W, self.b


def main():
    NN = Network([784, 10])
    NN.train(train_inputs[:, 0:10000], train_labels[0:10000], learning_rate=0.5)
    res = NN.predict(test_inputs)
    accuracy = np.sum(res.reshape(10000) == test_labels.reshape(10000), axis=0) / 10000.
    print "The accuracy is {0}".format(accuracy)


if __name__ == '__main__':
    main()
