"""
A3C For Hyperparameter Optimization

Modified from https://github.com/MorvanZhou/pytorch-A3C - MIT License
"""

# %%

from queue import Empty
import numpy as np
import torch
from torch import cuda
from torch._C import device
import torch.nn as nn
from utils import state_wrap, set_init, push_and_pull, record, GetMaxParam
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import math, os

from environment import HPOptEnv
from models.mnist_cnn import MNIST_CNN

from logger import Logger

from settings import *

os.environ["OMP_NUM_THREADS"] = "1"

# %%

env = HPOptEnv(MNIST_CNN, MAX_EP_STEP, GetMaxParam(MNIST_CNN))

N_S = GetMaxParam(MNIST_CNN)
N_A = env.action_space.shape[0]

del env

# %%

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 10)
        self.mu = nn.Linear(10, a_dim)
        self.sigma = nn.Linear(10, a_dim)
        self.c1 = nn.Linear(s_dim, 10)
        self.v = nn.Linear(10, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.MultivariateNormal
        self.from_id = torch.multiprocessing.Value('i')
        self.from_id.value = -1
        self.param_queue = torch.multiprocessing.Queue()

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        cov_n = sigma.diag_embed()
        # cov_n = torch.Tensor([[0 for _ in range(self.a_dim)]]).diag_embed()
        m = self.distribution(mu.view(size=(self.a_dim,)).data, cov_n)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        cov_n = sigma.diag_embed()
        m = self.distribution(mu, cov_n)
        log_prob = m.log_prob(a)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi)) * self.a_dim + cov_n.logdet()  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, sync_sema, sync_cond, global_states, global_params, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.index = name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.sync_sema = sync_sema
        self.sync_cond = sync_cond
        self.global_states = global_states
        self.global_params = global_params
        self.env = HPOptEnv(MNIST_CNN, MAX_EP_STEP, N_S, self.name)
        self.from_id = mp.Value('i')

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s : torch.Tensor = self.env.reset().cpu()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                s = state_wrap(s, N_S)
                a = self.lnet.choose_action(s)
                s_, r, done, _ = self.env.step(a.squeeze())
                s_ = s_.cpu()

                if self.name == 'w0':
                    self.env.render()
                
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # dropping
                    self.global_states[self.index] = s_.detach()
                    self.global_params[self.index] = self.env.model.model.state_dict()

                    Logger.Print("main", True, f"{self.name} Ready To Sync")
                    
                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    
                    self.sync_sema.release()

                    self.sync_cond.acquire()
                    self.sync_cond.wait()
                    
                    self.sync_cond.release()

                    if done:  # done and print information
                        break
                    
                    # check copy
                    if self.from_id.value != -1:
                        self.env.model.model.load_state_dict(self.global_params[self.from_id.value])
                        s_ : torch.Tensor = self.global_states[self.index]
                        self.from_id.value = -1
                
                s : torch.Tensor = s_
                total_step += 1

        self.res_queue.put(None)
    
    # load env from another worker
    def loadfrom(self, from_id):
        assert from_id != -1

        self.from_id.value = from_id

        # copy state
        self.global_states[self.index] = self.global_states[from_id]

    def save(self):
        pass
        
# %%

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    checkpoints = setup()

    torch.manual_seed(777)

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    print("State : ", N_S, "Action : ", N_A)

    GetMaxParam(MNIST_CNN, True) # Print Model Specification

    n_processes = N_WORKERS

    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing

    if not checkpoints is None:
        if "checkpoint" in checkpoints:
            gnet.load_state_dict(torch.load(os.path.join(DIR_CHECKPOINT, "checkpoint")))

    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue, global_states, global_params = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Manager().list([None for i in range(n_processes)]),  mp.Manager().list([None for i in range(n_processes)])

    sync_cond = mp.Condition(mp.Lock())

    sync_sema = [mp.Semaphore() for _ in range(n_processes)]
    for s in sync_sema:
        s.acquire()

    Logger.Print("main", True, "Initialization Complete")

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, sync_sema[i], sync_cond, global_states, global_params, i) for i in range(n_processes)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        # sync with workers
        [sync_sema[i].acquire() for i in range(n_processes)]

        Logger.Print("main", True, "Sync Complete. Starting Exploit Stage.")

        sync_cond.acquire()

        # calc values
       
        global_value = [gnet.forward(state_wrap(s, N_S))[-1].data.numpy()[0] for s in global_states]

        # do dropping task
        indexes = np.array(global_value).argsort()
        drop = indexes[:N_DROP]
        best = indexes[::-1][:N_DROP]
        
        for drop_idx, best_idx in zip(drop, best):
            Logger.Print("main", True, f"Drop w{drop_idx} : {global_value[drop_idx]} <--- {global_value[best_idx]} : w{best_idx} Best")
            workers[drop_idx].loadfrom(best_idx)

        # end of dropping

        Logger.Print("main", True, "Exploit Stage Complete.")

        torch.save(gnet.state_dict(), os.path.join(DIR_CHECKPOINT, "checkpoint"))

        Logger.Print("main", True, "Making Checkpoint OK.")

        sync_cond.notify_all()

        sync_cond.release()

        # collect reward
        try:
            r = res_queue.get_nowait()
            print(r)
            if r is not None:
                res.append(r)
            else:
                break
        except Empty:
            pass
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
# %%
