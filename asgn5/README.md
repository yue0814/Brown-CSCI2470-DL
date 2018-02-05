Assignment 5: Recurrent Neural Network Language Model
====
Yue Peng
----

In my first step, I initialized some hyperparameters following the instruction like batch size = 20, learning rate = 1e-3, and window size = 50. During debugging, I found out increasing the embedding size(1024) and LSTM size(600) did improve the model capacity. Weights here I chose another distribution following random uniform(-0.2, 0.2). Biases here I chose a small negative constant(-1.5) instead of zeros or truncated normal. 