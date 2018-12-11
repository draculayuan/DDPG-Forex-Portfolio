The model is a Forex Portfolio Management Tool based on Reinforcement Learning (Deep Deterministic Policy Gradient). The main structure is derived from yanpaulau's github DDPG-Keras-Torcs, https://github.com/yanpanlau/DDPG-Keras-Torcs. Please refer to the abovementioned github repo as well as https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html for the theory and intuition behind DDPG.

To run the code, type:
python ddpg.py --mode ?? --load ?? --e ??
in to command prompt/terminal to start.

Select mode from train or test (determine train or test)
Select load from true or false (if load existing weights)
Type any integer for e to set number of episode for train mode (for test mode it will be auto set to 1)
