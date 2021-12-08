from matplotlib import axes, pyplot as plt
from settings import *
import pandas as pd

class _bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

plt.ion()

class Logger:
    _owners = {'main', 'w0'}
    _plots = {}
    _plot_data = {}

    @staticmethod
    def Print(owner : str, positive : bool, *args) -> None:
        if positive:
            mark = _bcolors.OKBLUE + "+" + _bcolors.ENDC
        else:
            mark = _bcolors.FAIL + "-" + _bcolors.ENDC
        if owner in Logger._owners:
            owner = _bcolors.OKCYAN + owner + _bcolors.ENDC
            print(f"[{mark}] <{owner}> : {' '.join(map(lambda x: str(x), args))}\n")

    @staticmethod
    def UpdatePlot(owner : str, **kwargs):
        if owner not in Logger._owners:
            return
        if owner not in Logger._plots.keys():
            Logger._plots[owner] = plt.subplots(figsize=(8,6))
            
            figure, ax = Logger._plots[owner]
            ax.set_title(f"Progress of worker {owner}",fontsize=25)
            ax.set_xlabel("Step",fontsize=18)
            ax.set_ylabel("Model Loss",fontsize=18)
        
        figure, ax = Logger._plots[owner]
        if 'reset' in kwargs and kwargs['reset']:
            ax.cla()
        ax.scatter(kwargs['x'],kwargs['y'], color="blue")
        
        if owner not in Logger._plot_data:
            Logger._plot_data[owner] = ([], [])
        xdat, ydat = Logger._plot_data[owner]

        xdat.append(kwargs['x'])
        ydat.append(kwargs['y'])
        
        ax.plot(xdat, ydat, color="blue")
        
        figure.canvas.draw()
    
        figure.canvas.flush_events()