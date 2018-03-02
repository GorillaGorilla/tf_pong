
import tensorflow as tf
import numpy as np
import  agent
import cloudpickle as pickle
import gym
import matplotlib.pyplot as plt

learning_rate = 1e-1
gamma = 0.99  # discount factor for reward
D = 80*80
resume = False
render = False
# np.set_printoptions(threshold='nan')



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

    return I.astype(np.float).ravel()

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
save_location = 'models/pong_model_softmax.ckpt'

def actionsFromSoftmax(probs):
    # print(probs)
    return np.random.choice([1,2,3],1, p=probs)

    # random_num = np.random.uniform()
    # for i, prob in enumerate(probs):
    #     chosenValue = 2
    #     chosen = 0
    #     if (prob < random_num & prob < chosenValue):
    #         chosen = prob
    #         chosen = i
    # return chosen

with tf.Session() as sess:
    if resume:
        print('resuming')
        saver.restore(sess, save_location)
    else:
        sess.run(tf.global_variables_initializer())

    observation = env.reset()
    gradBuffer = sess.run(agent.tvars)
    gradBuffer = resetGradBuffer(gradBuffer)
    agent.setSession(sess)


    while running:
        if render: env.render()
        # keep running episodes
        cur_x = prepro(observation, count)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # if count % 10 == 0:
        #     print('count is 10')
        #     reshaped = x.reshape(80,80)
        #     plt.imshow(reshaped)
        #     plt.show()

        x = np.reshape(x, [1, D])
        xs.append(x)
        probs, logitProb, W2 = agent.evalPolWithIntermediates(x)
        action = actionsFromSoftmax(probs[0])
        y = [0,0,0]
        y[action - 1] = 1


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


            print('logitProb',logitProb)
            print('log logitProb', tf.log(logitProb))
            print('probs', probs)
            print('y', y)
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
            print(y_n)
            tGrad = agent.calculatePolicyGradients(x_n, y_n, grads)
            print('tGrad[0]',tGrad[0].shape)
            if np.sum(tGrad[0] == tGrad[0]) == 0:
                break
            # aggregate the gradients into the buffer (can just sum them as 2 variables, savesa  dimension)
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            # then train the policy network in a batch!
            if episode_number % 10 == 0:
                # weights1 = agent.writeWeights()
                # first1 = weights1[0][:, 0]
                # firstImg1 = first1.reshape([80, 80])
                # print('grad1', gradBuffer[0][:, 0])
                # print('weights[2].shape', weights1[1].shape)
                # print('weights1[1]',weights1[1])
                # print('grad2', gradBuffer[1])
                #
                # plt.subplot(211)
                # plt.imshow(gradBuffer[0][:, 0].reshape(80, 80),cmap='gray')
                # plt.subplot(212)
                # plt.imshow(gradBuffer[0][:, 1].reshape(80, 80),cmap='gray')
                # plt.show()
                # input("Press Enter to continue...")


                agent.trainPolicyNetwork(gradBuffer[0], gradBuffer[1])
                resetGradBuffer(gradBuffer)


                weights2 = agent.writeWeights()
                # first2 = weights2[0][:, 0]
                # firstImg2 = first2.reshape([80, 80])
                # img_diff = firstImg2 - firstImg1
                # print(img_diff)
                # plt.figure(1)
                # plt.imshow(firstImg1,cmap='gray')
                # plt.figure(2)
                # plt.imshow(firstImg2,cmap='gray')
                # plt.figure(3)
                # plt.imshow(img_diff,cmap='gray')
                # plt.show()
                # input("Press Enter to continue...")
                # print(firstImg2.shape)
                # print('weights2[2].shape', weights2[1].shape)
                # print('weights2[1]', weights2[1])
                # print('grad22', gradBuffer[1])

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            # if reward_sum != -21:
            #     point = {'x': x_n, 'y': y_n, 'r': rs_n}
            #     # running = False
            #     pickle.dump(point, open('winning_point.p', 'wb'))

            if episode_number % 100 == 0:
                model = {'W1': agent.getW1(), 'W2': agent.getW2()}
                # pickle.dump(agent.writeWeights(), open('nn_p_save.p', 'wb'))
                saver.save(sess, save_location)

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




