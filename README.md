# Brown-CSCI2470-DL

Over the past few years, Deep Learning has become a popular area, with deep neural network methods obtaining state-of-the-art results on applications in computer vision (Self-Driving Cars), natural language processing (Google Translate), and reinforcement learning (AlphaGo). 
This course intends to give students a practical understanding of the field of Deep Learning, through lectures and labs covering both the theory and application of neural networks to the above areas (and more!). We introduce students to the core concepts of Deep Neural Networks, including the backpropagation algorithm for training neural networks, as well as specific operations like convolution (in the context of computer vision), and word embeddings and recurrent neural networks (in the context of natural language processing). We also use the Tensorflow Framework for the expression of deep neural network models.

There were eight assignments written in Python and TensorFlow.

## HW1: MNIST Neural Network From Scratch
In this assignment, you will be coding a single-layer neural network to classify handwritten digits from scratch, using only NumPy. You will not be using TensorFlow, or any other deep learning framework


## HW2: MNIST Neural Network with Tensorflow
In this assignment, you will be improving the network from HW1 using TensorFlow.
First, you should rewrite the network from HW1 using TensorFlow. Then, to improve accuracy, you should add another hidden layer to the network before the softmax layer. Finally, you should experiment with the network's hyperparameters (batch size, learning rate & hidden layer size) to find those the give the best results.


## HW3: Convolutional Neural Networks
Goal:
We will be building a Convolutional Neural Network (CNN) with two convolution & max pooling layers for the MNIST digit classification task.


## HW4: Trigram Language Model
Goal:
We will be building a Trigram Language Model with Word Embeddings for language modeling the Penn Treebank Corpus.


## HW5: Recurrent Neural Network Language Model
Goal:
We will be building a Recurrent Neural Network Language Model with Word Embeddings for language modeling the Penn Treebank Corpus.


## HW6: Sequence to Sequence Machine Translation
Goal:
We will be building a simple set of models for performing Sequence-to-Sequence Machine Translation. Instead of building a full, complex machine translation pipeline, we will build a simple model that takes in short (< 12 word) French sentences, and outputs the English translations.

The first model we will build is a vanilla sequence-to-sequence model that uses an RNN encoder to encode French sentences, and a separate RNN Decoder to generate the translations. For this model, we will be passing the final hidden state of the Encoder as the initial state of the Decoder.

The second model we will build is a sequence-to-sequence model with "pseudo-attention." Instead of implementing the complex attention mechanisms discussed in class, we will instead be treating attention as a fixed table that, for each pair of English and French word positions, returns a corresponding weight.


## HW7: REINFORCE
Goal:
We will be implementing the REINFORCE Algorithm for the Cartpole-v0 task in OpenAI Gym. For this assignment, you will need to install the OpenAI Gym library (locally, it's already installed in the course virtual environment).

We recommend that during development, you render the OpenAI gym environment, every time you perform an action. While this significantly slows down your train time, it is good to see how your model is learning to perform a given task. When you turn your assignment in however, turn rendering off.


## HW8: Advantage Actor-Critic
Goal:
We will be implementing the Advantage Actor-Critic (A2C) Algorithm for the Cartpole-v1 (note difference from HW 7) task in OpenAI Gym. For this assignment, you will need to install the OpenAI Gym library (locally, it's already installed in the course virtual environment).

We recommend that during development, you render the OpenAI gym environment, every time you perform an action. While this significantly slows down your train time, it is good to see how your model is learning to perform a given task. When you turn your assignment in however, turn rendering off.