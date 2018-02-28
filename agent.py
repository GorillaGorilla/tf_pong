import tensorflow as tf
import numpy as np


class policy:
    def __init__(self, H, D, learning_rate):

        # define forward network
        self.observations = tf.placeholder(tf.float32, [None, D], name="frame_x")
        self.W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
        self.W2 = tf.get_variable("W2", shape=[H, 3], initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.matmul(layer1, self.W2)
        self.probability = tf.nn.softmax(self.logits)





        # training stuff
        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 3], name="input_y")
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
        batchGrad = [self.W1Grad, self.W2Grad]
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
        loss = tf.reduce_mean(tf.multiply(self.advantages,cross_ent))
        # loglik = tf.log(self.input_y * (self.input_y - self.probability) + (1 - self.input_y) * (self.input_y + self.probability))
        # loss = -tf.reduce_mean(loglik * self.advantages)
        self.newGrads = tf.gradients(loss, self.tvars)
        self.updateGrads = optimizer.apply_gradients(zip(batchGrad, self.tvars))

    def trainPolicyNetwork(self, W1Grad, W2Grad):
        self.sess.run(self.updateGrads, feed_dict={self.W1Grad: W1Grad, self.W2Grad: W2Grad})

    def calculatePolicyGradients(self, epx, epy, discounted_epr):
        newGrads= self.sess.run(self.newGrads, feed_dict={self.observations: epx, self.input_y: epy, self.advantages: discounted_epr})
        return newGrads

    def setSession(self, sess):
        self.sess = sess

    def getW1(self):
        return self.sess.run(self.W1)

    def getW2(self):
        return self.sess.run(self.W2)

    def evaluatePolicy(self, observations):
        tfprob = self.sess.run(self.probability, feed_dict={self.observations: observations})
        return tfprob

    def evalPolWithIntermediates(self, observations):
        tfprob, logitProb, W2 = self.sess.run([self.probability, self.logits, self.W2], feed_dict={self.observations: observations})
        return tfprob, logitProb, W2

    def writeWeights(self):
        weights = self.sess.run(self.tvars)
        print(weights)
        return str(weights)