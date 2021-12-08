# %%
from torch.optim import Adam
from environment import HPOptEnv
from models.mnist_cnn import MNIST_CNN
# %%
def GetMaxParam(model : type):
    env = HPOptEnv(model, 1)

    max_hyper = [env.hyper.range[n][1] for n in env.hyper.GetNames()]

    env.step(max_hyper)

    max_param = env.model.GetParameterSize()

    return max_param

# %%
env = HPOptEnv(MNIST_CNN, 10, GetMaxParam(MNIST_CNN), "main")

_, reward, done, _ = env.step(env.action_space.sample())

env.model.optimizer = Adam(env.model.model.parameters(), lr=0.001, weight_decay=0.01)

# %%

done = False
# %%
while(not done):
    _, reward, done, _ = env.step(env.action_space.sample())
    print(env.model.optimizer.param_groups[0]['lr'])
    env.render()
# %%
