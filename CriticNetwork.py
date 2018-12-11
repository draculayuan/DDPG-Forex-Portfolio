import numpy as np
import math
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Conv2D, Add
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
FILTERS = 8

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        #self.clipped_action_grads = [tf.clip_by_value(grad, -1., 1.) for grad in self.action_grads]
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size, 16, 1])  
        c0 = Conv2D(FILTERS, kernel_size=(3,4), strides=(1,1), padding='valid', activation='relu')(S)
        c1 = Conv2D(FILTERS, kernel_size=(3,4), strides=(1,1), padding='valid', activation='relu')(c0)
        f0 = Flatten()(c1)
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(f0)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        #h2 = merge([h1,a1],mode='sum') 
        h2 = Add()([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='relu')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

        return model, A, S 
