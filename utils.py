"""
Functions that use multiple times
"""

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import environment

from prettytable import PrettyTable

def Count_Parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# %%
def GetMaxParam(model : type, verbose : bool = False):
    env = environment.HPOptEnv(model, 1)

    max_hyper = [env.hyper.range[n][1] for n in env.hyper.GetNames()]

    env.step(max_hyper)

    max_param = env.model.GetParameterSize()

    if verbose:
        Count_Parameters(env.model.model)

    return max_param

def state_wrap(s : torch.Tensor, N_S):
    return F.pad(s, (0, N_S - s.shape[0]))

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.xavier_uniform(layer.weight)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    N_S = gnet.s_dim
    
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(state_wrap(s_, N_S))[-1].data.numpy()[0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        torch.stack(bs),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )