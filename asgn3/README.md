Homework 3: Convolutional Neural Networks
====
Yue Peng
----

In my first step, I initialized some hyperparameters following the instruction like batch size = 50, learning rate = 1e-4, padding = 'SAME' and epochs = 2000. As for the ksize and stride. I refer to the AlexNet which used stride=[1, 1, 1, 1] in conv2d and [1, 2, 2, 1] in maxpool2d. I set a hidden layer size as 625 and kept increasing with a small amount until the training accuracy and loss of mini batch looks great. Finally, the hidden layer size is 784.