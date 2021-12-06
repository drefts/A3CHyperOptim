from gym import spaces
import numpy as np
from torch import sigmoid
from torch import Tensor
class Hyperparameter:
    def __init__(self) -> None:
        self.params = {}
        self.range= {}
        self.discrete = {}
        self.fixed = {}
        pass
    
    def Register(self, name : str, value : float, param_range : tuple, discrete : bool, fixed : bool):
        assert len(param_range) == 2
        assert param_range[0] <= value <= param_range[1]
        
        self.params[name] = value
        self.range[name] = param_range
        self.discrete[name] = discrete

    def Get(self, name) -> float:
        if self.discrete[name] == True:
            return round(self.params[name])
        return self.params[name]

    def Set(self, name, value):
        assert name in self.params.keys()
        
        value = min(value, self.range[name][1])
        value = max(self.range[name][0], value)

        assert self.range[name][0] <= value <= self.range[name][1]
        
        self.params[name] = value
    
    def ApplyAction(self, action):
        for i, n in enumerate(self.GetNames()):
            self.Set(n, self.range[n][0] + sigmoid(Tensor([action[i]])).item() * (self.range[n][1] - self.range[n][0])) # apply action from mean

    def GetNames(self) -> list:
        return self.params.keys()

    def GenerateSpace(self):
        keys = self.params.keys()
        return spaces.Box(low=np.array([self.range[k][0] for k in keys]), high=np.array([self.range[k][1] for k in keys]), shape=(len(keys),), dtype=np.float32)

    def GetParameterString(self):
        from prettytable import PrettyTable
        table = PrettyTable(["Name", "Value"])
        for n in self.GetNames():
            table.add_row([n, self.params[n]])
        return table.get_string()