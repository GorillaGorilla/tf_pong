
import tensorflow as tf
import numpy as np
import  agent
import cloudpickle as pickle
import gym
import matplotlib.pyplot as plt

learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
D = 80
resume = False
save_url = './models/pong_model_4_lr3.ckpt'

# np.set_printoptions(threshold='nan')


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


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

def prepro(I, c):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    # if c % 10 == 0:
    #     plt.imshow(I)
    #     plt.show()
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # if c % 10 == 0:
    #     plt.imshow(I)
    #     plt.show()
    return I.astype(np.float)

def discount_epr(rs_n):
    discounted_epr = discount_rewards(rs_n)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    return discounted_epr

reward_sum = 0
running_reward = None
prev_x = None
env = gym.make('Pong-v0')
tf.reset_default_graph()
agent = agent.policy(200, D, learning_rate)
episode_number = 0
reward_sum = 0
running_total = 0
xs, ys, dlogps, rs = [], [], [], []
saver = tf.train.Saver()
count = 0
running = True



with tf.Session() as sess:
    if resume:
        print('resuming')
        saver.restore(sess, './models/pong_model_3_lr3.ckpt')
    else:
        sess.run(tf.global_variables_initializer())

    observation = env.reset()
    gradBuffer = sess.run(agent.tvars)
    gradBuffer = resetGradBuffer(gradBuffer)
    agent.setSession(sess)


    while running:
        # keep running episodes
        cur_x = prepro(observation, count)
        x = cur_x - prev_x if prev_x is not None else np.zeros((D,D))
        prev_x = cur_x

        # if count % 10 == 0:
        #     print('count is 10')
        #     reshaped = x.reshape(80,80)
        #     plt.imshow(reshaped)
        #     plt.show()

        x = np.reshape(x, [1, D, D, 1])


        xs.append(x)
        probability = agent.evaluatePolicy(x)
        action = 2 if np.random.uniform() < probability else 3  # fake label, what does this mean???
        y = 1 if action == 2 else 0
  # this is a regularisation gradient that pushes slighty for the thing that happened to happen if it was likely, and strongly for it to happen again if it was unlikely
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rs.append(reward)
        ys.append(y)
#         calculate next action from observation (get loss at this point?)
#         step environment
#         add observations, action, reward to arrays

        count += 1
        # print('episode %d reward_sum  %f' % (episode_number, reward))
        if done:
            episode_number += 1
            # numpyify these 2 arrays
            x_n = np.vstack(xs)
            y_n = np.vstack(ys)
            rs_n = np.vstack(rs)
            print(reward)

            if reward != -1:
                xpos_n = np.vstack(xs)
                ypos_n = np.vstack(ys)
                rposs_n = np.vstack(rs)


            xs, ys, dlogps, rs = [], [], [], []  # reset array memory for next point (in the game of pong)
            # then normalise, so take the mean and divide by standard deviation
            # do dlogps*rs now
            # if episode_number % 5 ==0:
            #     print('rs_n')
            #     print(rs_n)
            discounted_epr = discount_epr(rs_n)
            # if episode_number % 5 == 0:
            #     print('discounted_epr')
            #     print(discounted_epr)

            grads = discounted_epr
            # calculate the relevant gradients for the policy network
            tGrad = agent.calculatePolicyGradients(x_n, y_n, grads)

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

            # if reward_sum != -21:
            #     point = {'x': x_n, 'y': y_n, 'r': rs_n}
            #     # running = False
            #     pickle.dump(point, open('winning_point.p', 'wb'))

            if episode_number % 100 == 0:
                model = {'W1': agent.getW1(), 'W2': agent.getW2()}
                pickle.dump(agent.writeWeights(), open('nn_p_save2.p', 'wb'))
                saver.save(sess, save_url)

            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print(
                ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
