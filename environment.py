import gym
from gym import spaces
from numpy import array, dtype, float64
import torch
from abstractmodel import AbstractModel
from hyperparameter import Hyperparameter
from enum import Enum
from copy import copy
from torch.nn.utils import convert_parameters
import numpy as np
import pandas as pd

from logger import Logger

from settings import DIR_RESULT

class HPOptEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    class _EnvState(Enum):
        CLEAN = 0
        END = -1
    
    def __init__(self, model : type, max_epoch : int, max_params : int = None, name : str = ""):
        super(HPOptEnv, self).__init__()
        self.modeltype : type = model
        self.state : int = HPOptEnv._EnvState.CLEAN.value
        self.max_epoch : int= max_epoch
        self.max_params = max_params
        self.model : AbstractModel = self.modeltype(self.modeltype.HyperparameterSpecification(), name)
        self.hyper : Hyperparameter = self.model.GetHyperParameter()
        self.loss_buffer : list = []
        self.name = name
        #self.model.Build()
        self.action_space = self.hyper.GenerateSpace()
        if max_params == None:
            self.model.Build()
        self.observation_space = spaces.Dict({"theta": self.model.GenerateSpace(self.max_params), "lambda": self.hyper.GenerateSpace()})
        self.dataframe = pd.DataFrame({})
        self.rewards = []

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()        
        ob = torch.cat([convert_parameters.parameters_to_vector(self.model.model.parameters()).cpu(), torch.from_numpy(np.array(list(self.hyper.params.values()), dtype=np.float32))][::-1])
        episode_over = self.state == HPOptEnv._EnvState.END.value
        return ob, reward, episode_over, {}

    def reset(self):
        # save result
        if self.state == HPOptEnv._EnvState.END.value:
            import os
            # save episode result
            self.dataframe = self.dataframe.append(pd.DataFrame({"Loss":self.loss_buffer, "Rewards": self.rewards}))
            self.dataframe.to_csv(os.path.join(DIR_RESULT, f"{self.name}_progress.csv"))
        
        self.state = HPOptEnv._EnvState.CLEAN.value
        self.rewards = []
        self.loss_buffer = []
        self.model : AbstractModel = self.modeltype(self.modeltype.HyperparameterSpecification(), self.name)
        self.model.Build()
        self.hyper : Hyperparameter = self.model.GetHyperParameter()
        self.observation_space = spaces.Dict({"theta": self.model.GenerateSpace(self.max_params), "lambda": self.hyper.GenerateSpace()})

        return torch.cat([convert_parameters.parameters_to_vector(self.model.model.parameters()).cpu(), torch.from_numpy(np.array(list(self.hyper.params.values()), dtype=np.float32))][::-1])

    def render(self, mode='human', close=False):
        Logger.Print(self.name, True, "State", self.state)
        if len(self.rewards) > 0:
            Logger.Print(self.name, True, "Reward", self.rewards[-1])
        Logger.Print(self.name,True, f"Hyperparameter State\n{self.hyper.GetParameterString()}")

        if len(self.loss_buffer) > 0:
            Logger.UpdatePlot(self.name, x=self.state - 1, y=self.loss_buffer[-1])
            

    def _take_action(self, action):
        # Hyperparameter Action Apply
        # Unused Code
        """
        if self.state == HPOptEnv._EnvState.CLEAN.value:
            self.model : AbstractModel = self.modeltype(self.modeltype.HyperparameterSpecification(), self.name)

            self.model.GetHyperParameter().ApplyAction(action)
            
            self.model.Build()

            self.hyper : Hyperparameter = self.model.GetHyperParameter()
            self.action_space = self.hyper.GenerateSpace()
            self.observation_space = spaces.Dict({"theta": self.model.GenerateSpace(), "lambda": self.hyper.GenerateSpace()})

            self.state += 1
            return
        """

        self.model.GetHyperParameter().ApplyAction(action)
        
        self.loss_buffer.append(self.model.Train().item()) # Save Reward

        self.state += 1

    def _get_reward(self):
        """ Reward is given for XY. """
        if self.state == self.max_epoch:
            reward =  -self.model.Validate()[0].item()
            self.state = self._EnvState.END.value
        elif len(self.loss_buffer) <= 1:
            reward = 0
        else:
            reward = self.loss_buffer[-2]-self.loss_buffer[-1]
        self.rewards.append(reward)
        return reward
        