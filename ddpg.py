import argparse
from forex_env import ForexEnv
import numpy as np
#import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
from sklearn.preprocessing import StandardScaler
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from preprocess import get_data
import os

OU = OU()       #Ornstein-Uhlenbeck Process

def scaling(state):
    scl = StandardScaler()
    state1 = scl.fit_transform(state[:,:4,0])
    state2 = scl.fit_transform(state[:,4:8,0])
    state3 = scl.fit_transform(state[:,8:12,0])
    state4 = scl.fit_transform(state[:,12:16,0])
    scaled_state = np.hstack([state1, state2, state3, state4])
    scaled_state = scaled_state.reshape((state.shape[0],state.shape[1],state.shape[2]))
    return scaled_state

def process_action(action):
    
    for i in range(len(action[0])):
        if action[0][i] < 0:
            action[0][i] = 0
            
    Sum = np.sum(action[0])
    action[0][0] = action[0][0]/Sum
    action[0][1] = action[0][1]/Sum
    action[0][2] = action[0][2]/Sum
    action[0][3] = action[0][3]/Sum
        
    if Sum == 0:
        action[0][0] = 0.25
        action[0][1] = 0.25
        action[0][2] = 0.25
        action[0][3] = 0.25
    
    return action

def playGame(flag, data, load, episode_count):    #1 means Train, 0 means simply Run
    earn = 0
    lose = 0
    acc_reward = 0
    thres = 0
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 4  #action_dim = 2(2 weights)
    state_dim = 24  #Consists of 24 hours (current + past 23) of open, close, high, low rates + value of 4 currencies hold (in SGD/JPY), size = 24 x 4

    #np.random.seed(1337)

    EXPLORE = 24.
    if flag == True:
        max_steps = 100
    else:
        max_steps = 360
        
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    
    #Reward plotting
    reward_list = []
    #Tensorflow GPU optimization

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    
    
    
    # Calculate 
    # Generate environment
    env = ForexEnv(data, flag)

    #Now load the weight
    print("Now we load the weight")
    
    if load == True:
        try:
            actor.model.load_weights("weights/actormodel.h5")
            critic.model.load_weights("weights/criticmodel.h5")
            actor.target_model.load_weights("weights/actormodel.h5")
            critic.target_model.load_weights("weights/criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")
    
    print("FOREX Experiment Start.")
    for i in range(episode_count):
        step = 0

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        s_t = env.reset() #hstack of 6 hours data+value of currencies hold 
        
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            #feature scaling
            s_t = scaling(s_t)
            a_t_original = actor.model.predict(s_t.reshape((1, s_t.shape[0], s_t.shape[1], s_t.shape[2])))
            print("original action:")
            print(a_t_original)
            a_t_actual = process_action(a_t_original)
            
            if flag == True:
                indicator = 1
            else:
                indicator = 0
                
            noise_t[0][0] = indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.01 , 0.01, 0.01)#values yet to be tuned
            noise_t[0][1] = indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.01 , 0.01, 0.01)
            noise_t[0][2] = indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.01 , 0.01, 0.01)
            noise_t[0][3] = indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.01 , 0.01, 0.01)
  
            a_t[0][0] = a_t_actual[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_actual[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_actual[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_actual[0][3] + noise_t[0][3]
            print("actual action:")
            print(a_t)
            
            s_t1, r_t, done, val, threshold = env.step(a_t[0])
            s_t1 = scaling(s_t1)
            
            if(j == 0):
                thres = threshold
                print("threshold benchmark ", thres)
            
            buff.add(s_t, a_t_original[0], r_t, s_t1, done)      #Add replay buffer
            reward_list.append(r_t)
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (flag):  
                print("updating parameters!!!!!!!!!")
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                print("weightsbefore")
                w1 = actor.model.get_weights()
                print(w1[0])
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
                w2 = actor.model.get_weights()
                print("weightsafter")
                print(w2[0])

            total_reward += r_t
            s_t = s_t1
            
            acc_reward = acc_reward + r_t
            
            print("Episode", i, "Step", step, "Action", a_t_original, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done or j == max_steps - 1:
                print("episode finishes")
                if val > thres:
                    earn = earn + 1
                else:
                    lose = lose + 1
                print(earn, lose)
                break

        if np.mod(i+1, 3) == 0:
            if (flag):
                print("Now we save model")
                actor.model.save_weights("weights/actormodel.h5", overwrite=True)
                with open("weights/actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("weights/criticmodel.h5", overwrite=True)
                with open("weights/criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
    
    if flag == False:
        
        #saving rewards for plotting return
        reward_list = np.array(reward_list)
        np.save("reward_list"+str(episode_count)+".npy", reward_list)
        print("Reward Saved for Plotting")

    print("Finish, accumulated reward:")
    print(acc_reward)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--e", type=int, required=True)
    args = parser.parse_args()
    
    train, test = get_data()
    
    if args.mode == "train":
        flag = True
        episode = args.e
        data = train
    else:
        flag = False
        episode = 1
        data = test
        
    if args.load == "true":
        load = True
    else:
        load = False

    playGame(flag, data, load, episode)
