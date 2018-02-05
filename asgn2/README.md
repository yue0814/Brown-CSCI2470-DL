Homework 2: MNIST Neural Network with Tensorflow
====
Yue Peng
----

In my first step, I initialized some hyperparameters with relatively large value to obtain a high accuracy, like batch size = 128, hidden layer size = 625 and epochs = 1500. As for learning rate which will mainly affect the optimization process. Since it cannot be too large or too small, I set it up as 0.001(Using AdamOptimizer). Then I observed whether training loss keep going down so that I can increase the learning rate and prevent it from oscillation. Then I set learning rate as 0.0015. What's next was choosing a batch size. I kept increasing batch size to obtain a better result until the running time exceeded 120s. Thus, I got a batch size = 512. After a few adjustments, I get a set of parameters having an accuracy of 0.98 on my local machine.