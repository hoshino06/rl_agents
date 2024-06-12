""" DDPG based PIRL implementation with pytorch
        1. agentOptions
        2. pinnOptions
        3. PIRLagent
        4. trainOptions
        5.train
"""

__author__ = 'Hikaru Hoshino'
__email__ = 'hoshino@eng.u-hyogo.ac.jp'

import os
import numpy as np
import random
import copy
from collections import deque # double-ended que
from tqdm import tqdm  # progress bar
import torch
from   torch.utils.tensorboard import SummaryWriter

# Agent Options
def agentOptions(
        DISCOUNT            = 0.99, 
        CRITIC_OPTIMIZER    = None,
        ACTOR_OPTIMIZER     = None,
        REPLAY_MEMORY_SIZE  = 5_000,
        REPLAY_MEMORY_MIN   = 100,
        MINIBATCH_SIZE      = 16, 
        UPDATE_TARGET_EVERY = 5, 
        EPSILON_INIT        = 1,
        EPSILON_DECAY       = 0.998, 
        EPSILON_MIN         = 0.01,
        RESTART_EP          = None,
        TAU                 = 0.2
        ):
    
    agentOp = {
        'DISCOUNT'          : DISCOUNT,
        'ACTOR_OPTIMIZER'   : ACTOR_OPTIMIZER,
        'CRITIC_OPTIMIZER'  : CRITIC_OPTIMIZER,
        'REPLAY_MEMORY_SIZE': REPLAY_MEMORY_SIZE,
        'REPLAY_MEMORY_MIN' : REPLAY_MEMORY_MIN,
        'MINIBATCH_SIZE'    : MINIBATCH_SIZE, 
        'UPDATE_TARGET_EVERY':UPDATE_TARGET_EVERY, 
        'EPSILON_INIT'      : EPSILON_INIT,
        'EPSILON_DECAY'     : EPSILON_DECAY, 
        'EPSILON_MIN'       : EPSILON_MIN,
        'RESTART_EP'        : RESTART_EP,
        'TAU'               : TAU
        }
    
    return agentOp

# PINN Options
def pinnOptions(
        CONVECTION_MODEL,
        DIFFUSION_MODEL,
        SAMPLING_FUN, 
        HESSIAN_CALC    = True,
        ):

    pinnOp = {
        'CONVECTION_MODEL': CONVECTION_MODEL,
        'DIFFUSION_MODEL' : DIFFUSION_MODEL, 
        'SAMPLING_FUN'    : SAMPLING_FUN,
        'HESSIAN_CALC'    : HESSIAN_CALC,
        }

    return pinnOp


# DDPG Agent class
class DDPGagent:
    def __init__(self, actor, critic, agentOp): 

        # Agent Options
        self.actNum  = 3
        self.agentOp = agentOp
        
        #actor-networks
        self.actor     = actor
        self.actor_optimizer = agentOp['ACTOR_OPTIMIZER']
        
        # critic-networks
        self.critic     = critic
        self.critic_optimizer  = agentOp['CRITIC_OPTIMIZER']

        # Target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Replay Memory
        self.replay_memory = deque(maxlen=agentOp['REPLAY_MEMORY_SIZE'])

        # Initialization of variables
        # self.epsilon = agentOp['EPSILON_INIT'] if agentOp['RESTART_EP'] == None else max( self.agentOp['EPSILON_MIN'], agentOp['EPSILON_INIT']*np.power(agentOp['EPSILON_DECAY'], agentOp['RESTART_EP']))
        self.target_update_counter = 0
        self.terminate            = False
        self.last_logged_episode  = 0
        self.training_initialized = False
        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.actor(state)
    
    def get_action_with_noise(self, state):
        
        action = self.get_action(state)
        
        Mean = 0
        ActionSize = 1
        Standarddeviation = 0.1;
        w = Mean + np.random.randn(ActionSize)*Standarddeviation
        
        action_float = action.detach().numpy() + w
        action_idx = int(action_float)
        return action_idx
   
    def get_qs(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_values = self.actor(state)
        combined_input = torch.cat([state, action_values],dim=1)
        return self.critic(combined_input)
    

    def train_step(self, experience, is_episode_done):
        ########################
        # Update replay memory
        self.update_replay_memory(experience)

        if len(self.replay_memory) < self.agentOp['REPLAY_MEMORY_MIN']:
            return

        #####################################
        # Update critic
        #####################################        
        # Sample minibatch from experience memory
        minibatch = random.sample(self.replay_memory, self.agentOp['MINIBATCH_SIZE'])

        X = [] #入力データのリスト
        y = [] #ターゲットデータのリスト
        
        # Calculate target values
        for transition in minibatch:
            current_state, action, reward, new_state, is_done = transition
            
            # ターゲットアクターとターゲットネットワークを使って次状態(new_state)の価値関数を推定
            opt_action = self.target_actor(torch.tensor(new_state, dtype=torch.float32))
            new_state  = torch.tensor(new_state, dtype=torch.float32)
            combined_input = torch.cat([new_state,opt_action], dim=0)
            future_qs = self.target_critic(combined_input)
            
            # 現状態(current_state)のクリティックの推定を計算
            if is_done:
                new_q = torch.tensor(reward, dtype=torch.float32) 
            else:
                new_q = torch.tensor(reward, dtype=torch.float32) + self.agentOp['DISCOUNT'] * future_qs
            
            X.append(current_state)
            y.append(new_q.detach().item())
         
        
        X = np.array(X,dtype = np.float32)
        y = np.array(y,dtype = np.float32)
                 
        # Critic Loss function
        actions = np.array([transition[1] for transition in minibatch], dtype=np.float32)
        X_and_U = np.concatenate((X, actions.reshape(-1,1)), axis=1)
        y_pred  = self.critic( torch.from_numpy(X_and_U) )
        y_trgt  = torch.from_numpy(y).unsqueeze(1)
        loss_critic = torch.nn.functional.mse_loss(y_pred, y_trgt)          

        # Update critic network
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        ############################
        # Update actor network 
        ############################
        states = torch.tensor(np.array([transition[0] for transition in minibatch], dtype=np.float32))
        actions_pred = self.actor(states)
        critic_input = torch.cat([states, actions_pred], dim=1)
        
        loss_actor = -self.critic(critic_input).mean()
        
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        ############################
        # Update target network (もっと早くなるかも)
        ############################
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
              target_param.data.copy_(self.agentOp['TAU'] * param.data + (1 - self.agentOp['TAU']) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
              target_param.data.copy_(self.agentOp['TAU'] * param.data + (1 - self.agentOp['TAU']) * target_param.data)
         
            
    def load_weights(self, ckpt_dir, ckpt_idx=None):

        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError("Directory '{}' does not exist.".format(ckpt_dir))

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums   = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt  = f'/agent-{check_nums.max()}'  
            ckpt_path    = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError("Check point 'agent-{}' does not exist.".format(ckpt_idx))

        checkpoint = torch.load(ckpt_path)
        self.critic.load_state_dict(checkpoint['weights'])
        self.target_critic.load_state_dict(checkpoint['target-weights'])        
        self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path    


###################################################################################
# Learning Algorithm

def trainOptions(
        EPISODES      = 50, 
        LOG_DIR       = None,
        SHOW_PROGRESS = True,
        SAVE_AGENTS   = True,
        SAVE_FREQ     = 1,
        RESTART_EP    = None
        ):
    
    trainOp = {
        'EPISODES' : EPISODES, 
        'LOG_DIR'  : LOG_DIR,
        'SHOW_PROGRESS': SHOW_PROGRESS,
        'SAVE_AGENTS'  : SAVE_AGENTS,
        'SAVE_FREQ'    : SAVE_FREQ,
        'RESTART_EP'   : RESTART_EP
        }
        
    return trainOp


def each_episode(agent, env, trainOp): 
    
    #############################
    # Reset esisodic reward and state
    episode_reward = 0
    current_state = env.reset()

    episode_q0 = agent.get_qs(current_state).max().item() 

    ###############################
    # Iterate until episode ends
    is_done = False
    while not is_done:

        # get action
        action_idx = agent.get_action_with_noise(current_state)
       
        # make a step
        new_state, reward, is_done = env.step(action_idx)
        episode_reward += reward

        # train Q network
        experience = (current_state, action_idx, reward, new_state, is_done)
        agent.train_step( experience, is_done )

        # update current state
        current_state = new_state

    return episode_reward, episode_q0
    

def train(agent, env, trainOp):
    
    # Log file
    if trainOp['LOG_DIR']: 
        
        # For training stats
        summary_writer = SummaryWriter(log_dir=trainOp['LOG_DIR'])        

    start = 1 if trainOp['RESTART_EP'] == None else trainOp['RESTART_EP']
    # Iterate episodes
    if trainOp['SHOW_PROGRESS']:     
        iterator = tqdm(range(start+1, trainOp['EPISODES'] + 1), ascii=True, unit='episodes')
    else:
        iterator = range(start+1, trainOp['EPISODES'] + 1)

    for episode in iterator:

        ep_reward, ep_q0 = each_episode(agent, env, trainOp)

        if trainOp['LOG_DIR']: 
            summary_writer.add_scalar("Episode Reward", ep_reward, episode)
            summary_writer.add_scalar("Episode Q0",     ep_q0,     episode)

            summary_writer.flush()

            if trainOp['SAVE_AGENTS'] and episode % trainOp['SAVE_FREQ'] == 0:
                
                ckpt_path = trainOp['LOG_DIR'] + f'/agent-{episode}'
                torch.save({'weights':        agent.critic.state_dict(),
                            'target-weights': agent.target_actor.state_dict(),
                            'replay_memory':  agent.replay_memory}, 
                           ckpt_path)
                
    return 
