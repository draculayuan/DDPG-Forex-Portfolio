import numpy as np
import math
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dropout, Dense, Flatten, Input, merge, Lambda, Conv2D, LeakyReLU, Concatenate
import tensorflow as tf
import keras.backend as K
#K.set_learning_phase(1)
import os

HIDDEN1_UNITS = 64
HIDDEN2_UNITS = 128
HIDDEN3_UNITS = 64
HIDDEN4_UNITS = 16
FILTERS = 8

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)      
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
   
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the actor")
        S = Input(shape=[state_size, 16, 1])   
        c0 = Conv2D(FILTERS, kernel_size=(3,4), strides=(1,1), padding='valid', activation='relu')(S)
        c1 = Conv2D(FILTERS, kernel_size=(3,4), strides=(1,1), padding='valid', activation='relu')(c0)
        f0 = Flatten()(c1)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(f0)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        #h3 = Dense(HIDDEN4_UNITS, activation='relu')(h2)
        #h2 = Dropout(0.3)(h2)
        
        A1 = Dense(1,activation='relu')(h2)
        #A1 = LeakyReLU()(A1)
        A2 = Dense(1,activation='relu')(h2)
        #A2 = LeakyReLU()(A2)
        A3 = Dense(1,activation='relu')(h2)
        #A3 = LeakyReLU()(A3)
        A4 = Dense(1,activation='relu')(h2)
        #A4 = LeakyReLU()(A4)
        
        #A = merge([A1,A2,A3,A4], mode='concat')
        A = Concatenate()([A1,A2,A3,A4])
        model = Model(input=S,output=A)
        return model, model.trainable_weights, S

