import gym
import numpy as np
import tensorflow as tf

gamma = 0.9999
state = tf.placeholder(shape=[None, 4], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4, 8], 0, 0.01, dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([8], 0, 0.01, dtype=tf.float32))
hidden = tf.nn.relu(tf.matmul(state, W) + b1)
O = tf.Variable(tf.random_uniform([8, 2], 0, 0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([2], 0, 0.01, dtype=tf.float32))
output = tf.nn.softmax(tf.matmul(hidden, O) + b2)

rewards = tf.placeholder(shape=[None], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
indicies = tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]), indicies)
loss = -tf.reduce_sum(tf.log(actProbs)*rewards)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

totRs = []
totOutput = []
game = gym.make('CartPole-v0')
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
                if dn:
                    rUnits = [(gamma**i)*v for i,v in enumerate(r_hist)]
                    disRs = [sum(rUnits[n:])/(gamma**n) for n in range(len(rUnits))]
                    feed_dict = {state:st_hist, actions:a_hist, rewards:disRs}
                    sess.run(optimizer,feed_dict=feed_dict)
                    totRs.append(step)
                    break
    totOutput.append(np.mean(totRs[-100:]))

print "After 3 different trials, the mean reward collected over the last 100 episodes for each trial is %.2f" % (sum(totOutput)/3.)
