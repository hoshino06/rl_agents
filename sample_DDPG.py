# %%
""" Sample program of PIRL safety
        1. main_torch
        2. main_tensorflow
"""

__author__ = 'Hikaru Hoshino'
__email__ = 'hoshino@eng.u-hyogo.ac.jp'

# general packages
import numpy as np
import random
from datetime import datetime

# Environment
from rl_env.planer_env import PlanerEnv, convection_model, diffusion_model, sample_for_pinn

###############################################################################
# 1. Main func with pytorch
def main_training():

    from agent.ddpg  import DDPGagent, agentOptions,train, trainOptions, pinnOptions
    from torch import nn
    from torch.optim import Adam

    ###########################
    # Environment
    env = PlanerEnv()
    actNum = 1 #入力の次元
    obsNum = len(env.reset())

    ############################
    # Agent
    class CriticNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum+actNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32,1),
            )

        def forward(self, x):
            output = self.linear_stack(x)
            return output
    critic = CriticNetwork().to('cpu')

    class ActorNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            output = self.linear_stack(x)
            return output
    actor = ActorNetwork().to('cpu')

    agentOp = agentOptions(
        DISCOUNT=1,
        CRITIC_OPTIMIZER = Adam(critic.parameters(), lr=1e-5),
        ACTOR_OPTIMIZER = Adam(actor.parameters(), lr=1e-5),
        REPLAY_MEMORY_SIZE=5000,
        REPLAY_MEMORY_MIN=100,
        MINIBATCH_SIZE= 8,
)
    
 #    pinnOp = pinnOptions(
 #        CONVECTION_MODEL=convection_model,
 #        DIFFUSION_MODEL=diffusion_model,
 #        SAMPLING_FUN=sample_for_pinn,
 # )

    agent = DDPGagent(actor, critic, agentOp)

    ######################################
    # Training
    LOG_DIR = 'logs/test'+datetime.now().strftime('%m%d%H%M')

    trainOp = trainOptions(
        EPISODES=3000,
        SHOW_PROGRESS=True,
        LOG_DIR=LOG_DIR,
        SAVE_AGENTS=True,
        SAVE_FREQ=500,
    )
    train(agent, env, trainOp)

    return agent, env




###############################################################################
if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)

    agent, rl_env = main_training()
# -*- coding: utf-8 -*-

