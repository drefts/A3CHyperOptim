from typing import Sequence
from hyperparameter import Hyperparameter
from abc import ABCMeta, abstractmethod
from torch.nn import Module
from gym import spaces
import numpy as np
from torch.nn.utils import convert_parameters


class AbstractModel:
    def __init__(self, hyper : Hyperparameter, name : str = '') -> None:
        self.hyper = hyper # Hyperparameter Set
        self.model = None
        self.name = name
    
    @staticmethod
    def HyperparameterSpecification(): # Hyperparameter Specification
        pass

    @abstractmethod
    def Build(self): # Build Model
        pass
    
    @abstractmethod
    def Train(self) -> float: # Train Model (for each Step)
        pass
    
    @abstractmethod
    def Validate(self) -> tuple: # Validate Model
        pass
    
    @abstractmethod
    def Predict(self): # Make Prediction
        pass
    
    def GetParameterSize(self):
        self.model.parameters()
        return convert_parameters.parameters_to_vector(self.model.parameters()).size()[0]

    def GenerateSpace(self, param_space = None): # Make Parameter Space
        if param_space == None: param_space = convert_parameters.parameters_to_vector(self.model.parameters()).size()[0]
        return spaces.Box(low=np.array([-np.inf] * param_space), high=np.array([np.inf] * param_space), shape=(param_space,), dtype=np.float32)

    def GetState(self): # Get Model State theta
        return self.model.state_dict()
    
    def SetState(self, state_dict): # Set Model State theta
        self.model.load_state_dict(state_dict)

    def SetHyperParameter(self, hyper : Hyperparameter): # Set Model Hyperparameter Lambda
        self.hyper = hyper
    
    def GetHyperParameter(self) -> Hyperparameter: # Get Model Hyperparameter Lambda
        return self.hyper