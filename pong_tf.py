
import tensorflow as tf
import numpy as np
import cloudpickle as pickle
import gym

learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
D = 80*80


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class policy:
    def __init__(self, H, D):

        # define forward network
        self.observations = tf.placeholder(tf.float32, [None, D], name="frame_x")
        self.W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
        self.W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(layer1, self.W2)
        self.probability = tf.nn.sigmoid(score)





        # training stuff
        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
        batchGrad = [self.W1Grad, self.W2Grad]
        loglik = tf.log(self.input_y * (self.input_y - self.probability) + (1 - self.input_y) * (self.input_y + self.probability))
        loss = -tf.reduce_mean(loglik * self.advantages)
        self.newGrads = tf.gradients(loss, self.tvars)
        self.updateGrads = adam.apply_gradients(zip(batchGrad, self.tvars))

    def trainPolicyNetwork(self, W1Grad, W2Grad):
        sess.run(self.updateGrads, feed_dict={self.W1Grad: W1Grad, self.W2Grad: W2Grad})

    def calculatePolicyGradients(self, epx, epy, discounted_epr):
        newGrads= self.sess.run(self.newGrads, feed_dict={self.observations: epx, self.input_y: epy, self.advantages: discounted_epr})
        return newGrads

    def setSession(self, sess):
        self.sess = sess

    def evaluatePolicy(self, observations):
        tfprob = self.sess.run(self.probability, feed_dict={self.observations: observations})
        return tfprob

    def writeWeights(self):
        weights = self.sess.run(agent.tvars)
        print(weights)
        return str(weights)

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


reward_sum = 0
running_reward = None
prev_x = None
env = gym.make('Pong-v0')
print(D)
tf.reset_default_graph()
agent = policy(200, D)
init = tf.global_variables_initializer()
episode_number = 0
xs, ys, dlogps, rs = [], [], [], []

with tf.Session() as sess:
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(agent.tvars)
    gradBuffer = resetGradBuffer(gradBuffer)
    agent.setSession(sess)


    while True:
        # keep running episodes
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x


        x = np.reshape(x, [1, D])
        xs.append(x)
        probability = agent.evaluatePolicy(x)
        action = 2 if np.random.uniform() < probability else 3  # fake label, what does this mean???
        y = 1 if action == 2 else 0

        dlogps.append(y - probability)  # this is a regularisation gradient that pushes slighty for the thing that happened to happen if it was likely, and strongly for it to happen again if it was unlikely
        observation, reward, done, info = env.step(action)
        rs.append(reward)
        ys.append(y)
#         calculate next action from observation (get loss at this point?)
#         step environment
#         add observations, action, reward to arrays


        reward_sum += reward
        # print('episode %d reward_sum  %f' % (episode_number, reward))
        if done:
            episode_number += 1
            # numpyify these 2 arrays
            x_n = np.vstack(xs)
            y_n = np.vstack(ys)
            rs_n = np.vstack(rs)
            dlogps_n = np.vstack(dlogps)



            xs, ys, dlogps, rs = [], [], [], []  # reset array memory for next point (in the game of pong)
            # then normalise, so take the mean and divide by standard deviation
            # do dlogps*rs now
            discounted_epr = discount_rewards(rs_n)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            grads = discounted_epr * dlogps_n
            # calculate the relevant gradients for the policy network
            tGrad = agent.calculatePolicyGradients(x_n, y_n, grads)
            print('grad calculated')

            if np.sum(tGrad[0] == tGrad[0]) == 0:
                break
            # aggregate the gradients into the buffer (can just sum them as 2 variables, savesa  dimension)
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad


            # then train the policy network in a batch!
            if episode_number % 20 == 0:
                agent.trainPolicyNetwork(gradBuffer[0], gradBuffer[1])
                resetGradBuffer(gradBuffer)

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            if episode_number % 100 == 0: pickle.dump(agent.writeWeights(), open('nn_p_save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print(
                ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

#             grab the saves states per step, and stack them up
#             process the states by calculating a discounted reward
#             calculate losses here using cross entropy??
#             do gradient backprop calculation
#             add gradients gradient buffer
#             if the batch size has been hit, apply the gradients to the weights


#             record all the




