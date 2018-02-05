import tensorflow as tf 
import numpy as np
import math
import time
import sys

args = sys.argv

start = time.time()
with open(args[1]) as train:
    words = [word for line in train for word in line.strip().split()]
    
    #vocabCount = {} 
    #for i in words:
    #    if i not in vocabCount.keys():
    #        vocabCount[i] = 1
    #    else:
    #        vocabCount[i] += 1
    #rareword = []
    #for word, count in vocabCount.items():
    #    if count <= 10:
    #        rareword.append(word)
    #for i,word in enumerate(words):
    #    if word in rareword:
    #        words[i] = word.replace(word, '*UNK*')
    vocab = set(words)
    print 'Number of words: %d' % len(words)
    print 'Size of vocabulary: %d' % len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    # id_to_word = {i: w for i, w in enumerate(vocab)}
    data = [word_to_id[w] for w in words]


with open(args[2]) as development:
    dev_words = [word for line in development for word in line.strip().split()]
    for i, v in enumerate(dev_words):
        if v not in vocab:
            dev_words[i] = v.replace(v, '*UNK*')
    print 'Number of words: %d' % len(dev_words)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    dev_data = [word_to_id[w] for w in dev_words]


def get_mini_batch(batch_size, step):
    ws, ts = [], []
    i = step * batch_size
    for _ in range(batch_size):
        while i < (step+1)*batch_size:
            w, t = (data[i:i+2], data[i+2])
            ws.append(w)
            ts.append(t)
            i += 1 
    return np.array(ws), np.array(ts)


def dev_get_mini_batch(batch_size, dev_step):
    ws, ts = [], []
    i = dev_step * batch_size
    for _ in range(batch_size):
        while i < (dev_step+1)*batch_size:
            w, t = (dev_data[i:i+2], dev_data[i+2])
            ws.append(w)
            ts.append(t)
            i += 1 
    return np.array(ws), np.array(ts)

end1 = time.time() - start
print end1

batch_size = 20
learning_rate = 1e-4
vocabSz = len(vocab)
embedSz = 100
hidden_size = 128 
# hidden_size = 625
inpt = tf.placeholder(tf.int32, shape=[batch_size])
inpt2 = tf.placeholder(tf.int32, shape=[batch_size])
answr = tf.placeholder(tf.int32, shape=[batch_size])
prob = tf.placeholder(tf.float32)
E = tf.Variable(tf.truncated_normal([vocabSz, embedSz], stddev=0.1))
embed = tf.nn.embedding_lookup(E, inpt)
embed2 = tf.nn.embedding_lookup(E, inpt2)
both = tf.concat([embed, embed2], 1)

W = tf.Variable(tf.truncated_normal([200, hidden_size], stddev=0.1))
b = tf.Variable(tf.truncated_normal([vocabSz], stddev=0.1))
U = tf.Variable(tf.truncated_normal([hidden_size, vocabSz], stddev=0.1))
d = tf.Variable(tf.truncated_normal([hidden_size],stddev=0.1))
A1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(both, W) + d), keep_prob=prob)
logits = tf.matmul(A1, U) + b  
# Loss
xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answr)
loss = tf.reduce_sum(xEnt)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    total_loss = 0
    while step <= ((len(words)-2)//batch_size)-1:
        w, t = get_mini_batch(20, step)
	c, _ = sess.run([loss, optimizer], feed_dict={inpt:w[:,0], inpt2:w[:,1], answr:t, prob:0.98})
        step += 1
        total_loss += c
        if step % 10000 == 0:
            print 'average loss until step %d: %.3f' % (step, total_loss/step)
    print "Optimization Completed"
    print 'The perplexity of training corpus is %.3f' % (math.exp(total_loss/len(words)))
    
    dev_step = 0
    total_loss = 0
    while dev_step <= ((len(dev_words)-2)//batch_size)-1:
        w, t = dev_get_mini_batch(20, dev_step)
	c = sess.run(loss, feed_dict={inpt:w[:,0], inpt2:w[:,1], answr:t, prob:1.0})
        dev_step += 1
        total_loss += c
    print 'The perplexity of dev corpus is %.3f' % (math.exp(total_loss/len(dev_words)))
end = time.time()
print 'Running time is {0}s'.format(round(end - start,2))

