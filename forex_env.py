import numpy as np
import pandas as pd



class ForexEnv:
    

    def __init__(self, data, flag):
        self.dataset = data
        self.initial_run = True
        self.time_index = 0
        self.init_val_currency_hold = np.array([100000, 100000, 100000, 100000])  #hardcoded for 4 currency-pairs and set the initial value to 100000)
        self.done = False
        self.flag = flag
        self.seed = 0

    def step(self, action):
        #calculate current portfolio value:
        #get current close exchange rate
        
        current_portfolio_val = self.cal_portfolio_value(self.time_index, self.init_val_currency_hold)
        record_val = current_portfolio_val
        print("current portfolio value is \n") 
        print(current_portfolio_val)
        
        #calculate new value of currency hold
        current_close_rate = self.dataset[self.time_index,[3,7,11,15]]
        new_val_currency_hold = current_portfolio_val * action * current_close_rate
        print("new value currency hold is \n") 
        print(new_val_currency_hold)
        
        #calculate value after one time interval
        future_portfolio_val = self.cal_portfolio_value(self.time_index+1, new_val_currency_hold)
        print("future portfolio value is \n") 
        print(future_portfolio_val)
        
        #calculate reward
        reward = future_portfolio_val - current_portfolio_val
        
        #check if done
        if future_portfolio_val <=10:
            self.done = True
            
        #compile states for return  (6 hours) from t-4 to t+1
        new_state = self.compile_state(self.time_index+1)
        #new_state = np.hstack([new_state, new_val_currency_hold.reshape((1,2))])   #changing dimension to 48
        
        #update initial value currency hold and time_index for next input
        self.init_val_currency_hold = new_val_currency_hold 
        self.time_index = self.time_index+1
        
        return new_state, reward, self.done, future_portfolio_val, record_val
    
    def cal_portfolio_value(self, t, v):
        
        close_rate = self.dataset[t,[3,7,11,15]]
        portfolio_value = v / close_rate
        portfolio_value = sum(portfolio_value)
        
        return portfolio_value
        
        
    def compile_state(self, t):
        state = self.dataset[t-23:t+1]
        state = state.reshape((state.shape[0], state.shape[1], 1))
        return state
        
    def reset(self):
        if self.flag == True:
            self.time_index = np.random.randint(30, 1000)
        else:
            self.time_index = 30
        print("time index: ", self.time_index)
        self.done = False
        #use initial value
        self.init_val_currency_hold = np.array([100000, 100000, 100000, 100000])
        
        state = self.compile_state(self.time_index)
        return state 
        
    
        

            
        
        
        
