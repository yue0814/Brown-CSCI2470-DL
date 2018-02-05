import gym
import numpy as np
import tensorflow as tf

gamma = 0.99
state = tf.placeholder(shape=[None, 4], dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([4, 8], 0, 0.01, dtype=tf.float32))
# b1 = tf.Variable(tf.random_uniform([8], 0, 0.01, dtype=tf.float32))
# hidden = tf.nn.relu(tf.matmul(state, W) + b1)


V1 = tf.Variable(tf.random_normal([4, 8], stddev=0.1, dtype=tf.float32))
b1 = tf.Variable(tf.random_normal([8], stddev=0.1, dtype=tf.float32))
v1Out = tf.nn.relu(tf.matmul(state, V1) + b1)
V2 = tf.Variable(tf.random_normal([8, 1], stddev=0.1, dtype=tf.float32))
b2 = tf.Variable(tf.random_normal([1], stddev=0.1, dtype=tf.float32))
vOut = tf.matmul(v1Out, V2) + b2

O = tf.Variable(tf.random_uniform([8, 2], 0, 0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([2], 0, 0.01, dtype=tf.float32))
output = tf.nn.softmax(tf.matmul(v1Out, O) + b2)

rewards = tf.placeholder(shape=[None], dtype=tf.float32)
disRewards = tf.placeholder(shape=[None], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
# critic loss
vLoss = tf.reduce_mean(tf.square(rewards - vOut))
indicies = tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]), indicies)
loss = -tf.reduce_sum(tf.log(actProbs)*disRewards)
loss = loss + vLoss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

totRs = []
totOutput = []
game = gym.make('CartPole-v1')
for n in range(3): 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        totRs = []
        for i in range(1000):
            st = game.reset()
            st_hist = []
            a_hist = []
            r_hist = []
            for step in range(999):
                # game.render()
                actDist = sess.run(output, feed_dict={state: [st]})
                act = np.random.choice(2, 1, p=actDist[0])[0]
                st1, r, dn, _ = game.step(act)
                st_hist.append(st)
                a_hist.append(act)
                r_hist.append(r)
                st = st1
                if step % 50 == 0 and step != 0 and not dn:
                    rUnits = [(gamma**i)*v for i,v in enumerate(r_hist)]
                    disRs = [sum(rUnits[n:])/(gamma**n) for n in range(len(rUnits))]
                    feed_dict = {state:st_hist, actions:a_hist, rewards:r_hist ,disRewards:disRs}
                    sess.run(optimizer,feed_dict=feed_dict)
                elif dn:
                    rUnits = [(gamma**i)*v for i,v in enumerate(r_hist)]
                    disRs = [sum(rUnits[n:])/(gamma**n) for n in range(len(rUnits))]
                    feed_dict = {state:st_hist, actions:a_hist,rewards:r_hist ,disRewards:disRs}
                    sess.run(optimizer,feed_dict=feed_dict)
                    totRs.append(step)
                    break
    totOutput.append(np.mean(totRs[-100:]))

print "After 3 different trials, the mean reward collected over the last 100 episodes for each trial is %.2f" % (sum(totOutput)/3.)
