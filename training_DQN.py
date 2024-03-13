# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random
from datetime import datetime

# keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# DQN agent
from agent.agent_DQN import RLagent, agentOptions, train, trainOptions

###################################################################################
# Environment (exmaple in ACC2024 paper)
class PlanerEnv:

    dt = 0.1;
    ACTIONS = [-1, 0, 1]       
    
    def reset(self):

        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.5  #sign(r(2))*0.8 + sign(r(3))*0.35*rand;
        T   = np.random.randint(20)*self.dt
        self.state = np.array([X1, X2, T])
        
        return self.state

    def step(self, action_idx):       
        
        # Current state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        
        # New state
        U  = self.ACTIONS[action_idx]
        new_state = np.array([
                          X1 + self.dt*( -X1**3- X2),
                          X2 + self.dt*( X1   + X2  +U),
                           T - self.dt 
                     ])
        
        # Check terminal conditios 
        isTimeOver = (T <= self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe

        # Reward
        if done and (not isUnsafe):
            reward = 1
        else:
            reward = 0

        self.state = new_state
        
        return new_state, reward, done


################################################################################################
# Main
###############################################################################
if __name__ == '__main__':

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)


    ###########################
    # Environment
    env    = PlanerEnv()
    actNum = len(env.ACTIONS)
    obsNum = len(env.reset())

    ############################
    # Agent    
    model = Sequential([
                Dense(32, input_shape=[obsNum, ]),
                Activation('tanh'), 
                Dense(32),  
                Activation('tanh'), 
                Dense(actNum),  
            ])
    
    agentOp = agentOptions(
        DISCOUNT   = 1, 
        LEARN_RATE = 1e-3, 
        REPLAY_MEMORY_SIZE = 5000, 
        )
    agent  = RLagent(model, actNum, agentOp)


    ######################################
    # Training

    #LOG_DIR = None
    LOG_DIR = 'logs/test'+datetime.now().strftime('%m%d%H%M')
    
    trainOp = trainOptions(
        EPISODES = 50, 
        SHOW_PROGRESS = True, 
        LOG_DIR     = LOG_DIR,
        SAVE_AGENTS = True, 
        SAVE_FREQ   = 10,
        )
    train(agent, env, trainOp)
    