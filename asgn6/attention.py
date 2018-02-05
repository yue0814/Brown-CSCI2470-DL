import tensorflow as tf
import numpy as np
import copy
import time
import sys

args = sys.argv

start = time.time()
###################### Preprocessing with "french_train.txt"
with open(args[1]) as f:
    sentences0 = [l.split() for l in f]
    max_length = max(map(lambda x:len(x), sentences0))
    print "The longest sentence in french_train.txt has %d words." % max_length

# Padding with STOP
french_train = copy.deepcopy(sentences0)
for s in french_train:
    s.extend(['STOP']*(13-len(s)))

######################## Preprocessing with "french_test.txt"
with open(args[3]) as f:
    sentences1 = [l.split() for l in f]
    max_length = max(map(lambda x:len(x), sentences1))
    print "The longest sentence in french_test.txt has %d words." % max_length

# Padding with STOP
french_test = copy.deepcopy(sentences1)
for s in french_test:
    s.extend(['STOP']*(13-len(s)))

######################### Preprocessing with "english_train.txt"
with open(args[2]) as f:
    sentences2 = [l.split() for l in f]
    max_length = max(map(lambda x:len(x), sentences2))
    print "The longest sentence in english_train.txt has %d words." % max_length

# Padding with STOP
english_train = copy.deepcopy(sentences2)
for s in english_train:
    s.extend(['STOP']*(13-len(s)))

##################### Preprocessing with "english_test.txt"
with open(args[4]) as f:
    sentences3 = [l.split() for l in f]
    max_length = max(map(lambda x:len(x), sentences3))
    print "The longest sentence in english_test.txt has %d words." % max_length

# Padding with STOP
english_test = copy.deepcopy(sentences3)
for s in english_test:
    s.extend(['STOP']*(13-len(s)))


## Vocabulary for French
wordsF = [word for line in french_train for word in line]
vocabFrench = set(wordsF)
print 'Size of French vocabulary: %d' % len(vocabFrench)
word_to_idF = {w: i for i, w in enumerate(vocabFrench)}
id_to_wordF = {i: w for i, w in enumerate(vocabFrench)}
datF_train = [word_to_idF[w] for w in wordsF]


## Vocabulary for English
wordsE = [word for line in english_train for word in line]
vocabEnglish = set(wordsE)
print 'Size of English vocabulary: %d' % len(vocabEnglish)
word_to_idE = {w: i for i, w in enumerate(vocabEnglish)}
id_to_wordE = {i: w for i, w in enumerate(vocabEnglish)}
datE_train = [word_to_idE[w] for w in wordsE]



wordsFt = [word for line in french_test for word in line]
for i, v in enumerate(wordsFt):
    if v not in vocabFrench:
        wordsFt[i] = v.replace(v, '*UNK*')
print 'Number of words in french_test.txt: %d' % len(wordsFt)
datF_test = [word_to_idF[w] for w in wordsFt]


wordsEt = [word for line in english_test for word in line]
for i, v in enumerate(wordsEt):
    if v not in vocabEnglish:
        wordsEt[i] = v.replace(v, '*UNK*')
print 'Number of words in english_test.txt: %d' % len(wordsEt)
datE_test = [word_to_idE[w] for w in wordsEt]    


def get_batches(fTrain, eTrain, BATCH_SZ, WINDOW_SZ):
    fTrain = np.array(fTrain,dtype=np.int32).reshape([-1, 13])
    eTrain = np.array(eTrain,dtype=np.int32).reshape([-1, 13])
    n_batches = int(fTrain.shape[0]/BATCH_SZ)
    fTrain = fTrain[0:n_batches*BATCH_SZ, :]
    eTrain = eTrain[0:n_batches*BATCH_SZ, :]
    for n in range(0, fTrain.shape[0], BATCH_SZ):
        x = fTrain[n: (n+BATCH_SZ), 0:WINDOW_SZ]
        y = eTrain[n:(n+BATCH_SZ),0:WINDOW_SZ]
        yield x, y



wSz = 12
bSz = 20
embedSz = 30
vfSz = len(vocabFrench)
veSz = len(vocabEnglish)
rnnSz = 64
LR = 1e-3
pad = np.array([word_to_idE['STOP']]*20, dtype=np.int32)
pad = pad.reshape([20,1])


encIn = tf.placeholder(tf.int32, shape=[bSz, wSz])
decIn = tf.placeholder(tf.int32, shape=[bSz, wSz])
ans = tf.placeholder(tf.int32, shape=[bSz, wSz])
keepPrb = tf.placeholder(tf.float32)
mask0 = tf.placeholder(tf.int32, shape=[bSz])
mask1 = tf.placeholder(tf.int32, shape=[bSz])
mask2 = tf.placeholder(tf.int32, shape=[bSz])

#W_att = tf.Variable(tf.truncated_normal([wSz, wSz], stddev=0.1))
#W_att = tf.Variable(np.ones([wSz, wSz],dtype=np.float32)*1./wSz)
W_att = tf.Variable(np.ones([wSz, wSz], dtype=np.float32))
W_att = tf.divide(W_att, tf.reduce_sum(W_att,axis=0))

with tf.variable_scope("enc"):
    F = tf.Variable(tf.truncated_normal((vfSz, embedSz), stddev=0.1))
    embs = tf.nn.embedding_lookup(F, encIn)
    embs = tf.nn.dropout(embs, keepPrb)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    initState = cell.zero_state(bSz, tf.float32)
    encOut, encState = tf.nn.dynamic_rnn(cell, embs,\
    sequence_length=mask0, initial_state=initState)
    encOut = tf.transpose(encOut,[0, 2, 1])
    decIT = tf.tensordot(encOut, W_att,[[2], [0]])
    decI = tf.transpose(decIT, [0, 2, 1])


with tf.variable_scope("dec"):
    E = tf.Variable(tf.truncated_normal((veSz, embedSz), stddev=0.1))
    embs = tf.nn.embedding_lookup(E, decIn)  
    embs_ext = tf.concat([embs, decI], 2)
    embs_ext = tf.nn.dropout(embs_ext, keepPrb)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    initState = cell.zero_state(bSz, tf.float32)
    decOut, _ = tf.nn.dynamic_rnn(cell, embs_ext,\
    sequence_length=mask1, initial_state=initState)
    

W = tf.Variable(tf.truncated_normal([rnnSz, veSz], stddev=0.1))
b = tf.Variable(tf.truncated_normal([veSz], stddev=0.1))
logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b
weights = tf.sequence_mask(mask2, wSz, dtype=tf.float32)
loss = tf.contrib.seq2seq.sequence_loss(logits, ans, weights)
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
prob = tf.nn.softmax(logits)
pred = tf.argmax(prob, axis=2)


def accuracy(words, y):
    acc = []  
    for i in range(bSz):
        accCounter = 0.
        rowCounter = 0
        flag = 0
        for j in range(wSz):
            if y[i,j] != word_to_idE['STOP']:
                rowCounter += 1
                if y[i,j] == words[i,j]:
                    accCounter += 1
            else:
                flag += 1
                if flag != 2:
                    rowCounter += 1
                    if y[i,j] == words[i,j]:
                        accCounter += 1
                else:
                    acc.append(accCounter/rowCounter)
                    break
    return acc
            

def secondTrueF(x):
    mask = []
    for i in range(bSz):
        flag = 0
        for j in range(wSz):
            if x[i,j] == word_to_idF["STOP"]:
                flag += 1
            if flag == 2 or j == wSz-1:
                mask.append(j)
                break
    mask = np.array(mask, dtype=np.int32)
    mask = mask.reshape([bSz])
    return mask


def secondTrueE(x):
    mask = []
    for i in range(bSz):
        flag = 0
        for j in range(wSz):
            if x[i,j] == word_to_idE["STOP"]:
                flag += 1
            if flag == 2 or j == wSz-1:
                mask.append(j)
                break
    mask = np.array(mask, dtype=np.int32)
    mask = mask.reshape([bSz])
    return mask


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    total = 0
    for x, y in get_batches(datF_train, datE_train, bSz, wSz):
        counter += 1
        batch_loss, _ = sess.run([loss, optimizer],\
                    feed_dict={encIn:x, decIn:np.hstack((pad,y[:,0:(wSz-1)])), ans:y, keepPrb:1.0,\
                    mask0:secondTrueF(x), mask1:secondTrueE(np.hstack((pad,y[:,0:(wSz-1)]))), mask2:secondTrueF(x)})
        total += batch_loss
        if counter % 1000 == 0:
            print "Average loss is %.3f " % (total/counter)
    print "Optimization Completed"
    acc = 0
    counter = 0
    for x, y in get_batches(datF_test, datE_test, bSz, wSz):
        # words (bSz, wSz)
        words = sess.run(pred,\
                    feed_dict={encIn:x, decIn:np.hstack((pad,y[:,0:(wSz-1)])) , ans:y, keepPrb:1.0,\
                    mask0:secondTrueF(x), mask1:secondTrueE(np.hstack((pad,y[:,0:(wSz-1)]))), mask2:secondTrueF(x)})  
        acc += sum(accuracy(words, y))/20
        counter += 1 

    print "The per symbol test accuracy is %.3f" % (acc/counter)
        
end = time.time()
print "The whole running time is %.2f" %(end - start)