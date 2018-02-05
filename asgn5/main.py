import tensorflow as tf
import numpy as np
import math
import sys
import time


args = sys.argv
start = time.time()
with open(args[1]) as train:
    words = [word for line in train for word in line.strip().split()]
    vocab = set(words)
    print 'Size of vocabulary: %d' % len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    dat = [word_to_id[w] for w in words]

with open(args[2]) as development:
    dev_words = [word for line in development for word in line.strip().split()]
    for i, v in enumerate(dev_words):
        if v not in vocab:
            dev_words[i] = v.replace(v, '*UNK*')
    print 'Number of words in dev.txt: %d' % len(dev_words)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    dev_dat = [word_to_id[w] for w in dev_words]
end1 = time.time()
print "Successfully loaded train.txt and dev.txt"
print "The loading time is %.3f" %(end1-start)
data = np.array(dat).reshape([-1,1])
dev_data = np.array(dev_dat).reshape([-1,1])

def get_batches(arr, BATCH_SZ, WINDOW_SZ):
    n_batches = int(len(arr)/(BATCH_SZ*WINDOW_SZ))
    arr = arr[:BATCH_SZ*n_batches*WINDOW_SZ]
    # (BATCH_SZ, n_batches)
    arr =  arr.reshape([BATCH_SZ, -1])
    # (BATCH_SZ, WINDOW_SZ)
    for n in range(0, arr.shape[1], WINDOW_SZ):
        x = arr[:, n: (n+WINDOW_SZ)]
        y = np.zeros_like(x)
	if n+WINDOW_SZ+1 > arr.shape[1]:
		z = arr[:,0]
	else:
		z = arr[:,n+WINDOW_SZ+1]
        y[:,:-1], y[:,-1] = x[:, 1:], z
	#y[:,:-1], y[:,-1] = x[:, 1:], y[:,0]
        yield x, y

WINDOW_SZ = 20
BATCH_SZ = 50
EMBED_SZ = 1024 
STATE_SZ = len(vocab)
LSTM_SZ = 600 
LR = 1e-3
#grad_clip = 5 

inputs = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
targets = tf.placeholder(tf.int32, shape=[BATCH_SZ, WINDOW_SZ])
weights=tf.placeholder(dtype=tf.float32,shape=[BATCH_SZ, WINDOW_SZ])
keep_prob = tf.placeholder(tf.float32)
E = tf.Variable(tf.truncated_normal([STATE_SZ, EMBED_SZ], stddev=0.1))
#should have shape=[BATCH_SZ, WINDOW_SZ, EMBED_SZ]
E_inputs = tf.nn.embedding_lookup(E, inputs)
train_weights=np.ones(shape=[BATCH_SZ, WINDOW_SZ],dtype=np.float32)

# LSTM
cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SZ, forget_bias=1.0)
drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
init_state = drop.zero_state(BATCH_SZ, tf.float32)

'''
outputs = []
with tf.variable_scope("RNN"):
    for time_step in range(WINDOW_SZ):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
        (cell_output, state) = cell(E_inputs[:, time_step,:], state)
        outputs.append(cell_output)
outputs.append(output)
'''  
output, _ = tf.nn.dynamic_rnn(drop, E_inputs, initial_state=init_state)


#seq_output = tf.concat(outputs, 1)
# (BATCH_SZ*WINDOW_SZ, LSTM_SZ)
#x = tf.reshape(seq_output, [-1, LSTM_SZ])
x = tf.reshape(output, [-1, LSTM_SZ])


#softmax_w = tf.Variable(tf.truncated_normal([LSTM_SZ, STATE_SZ],stddev=0.1)) 
softmax_w = tf.Variable(tf.random_uniform([LSTM_SZ, STATE_SZ],-0.2, 0.2)) 
#softmax_b = tf.Variable(tf.zeros([STATE_SZ]))
softmax_b = tf.Variable(tf.constant(-1.5,shape=[STATE_SZ]))
#softmax_b = tf.Variable(tf.truncated_normal(shape=[STATE_SZ],stddev=0.1))
# (BATCH_SZ*WINDOW_SZ, STATE_SZ)
logits = tf.matmul(x, softmax_w) + softmax_b
logits = tf.reshape(logits,[BATCH_SZ, WINDOW_SZ, STATE_SZ])
# final_state = state
# [BATCH_SZ, WINDOW_SZ, ]
loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=weights)
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
'''
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
train_op = tf.train.AdamOptimizer(LR)
optimizer = train_op.apply_gradients(zip(grads, tvars))
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_state)
    counter = 0
    total = 0
    for x, y in get_batches(data, BATCH_SZ, WINDOW_SZ):
        counter += 1
        batch_loss, _ = sess.run([loss, optimizer],\
			feed_dict={inputs:x, targets:y, weights:train_weights, keep_prob:1.0})
        total += batch_loss*WINDOW_SZ*BATCH_SZ
        if counter % 100 == 0:
            print "Average loss is %.3f " % (total/counter)
    print "Optimization Completed and the perplexity of train corpus is %.3f"\
         %(math.exp(total/(int(len(words)/(BATCH_SZ*WINDOW_SZ))*BATCH_SZ*WINDOW_SZ)))
    total_loss = 0
    for x, y in get_batches(dev_data, BATCH_SZ, WINDOW_SZ):    
        batch_loss = sess.run(loss,\
			feed_dict={inputs:x, targets:y, weights:train_weights, keep_prob:1.0})
        total_loss += batch_loss*BATCH_SZ*WINDOW_SZ
    print 'The perplexity of dev corpus is %.3f' % (math.exp(total_loss/(int(len(dev_words)/(BATCH_SZ*WINDOW_SZ))*BATCH_SZ*WINDOW_SZ)))
end2 = time.time()
print "The whole performance time is %.2f seconds" %(end2 - start)
