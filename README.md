# A3CHyperOptim

`main.py` trains the given environment using A3C algorithm.

`abstractmodel.py` defines the template for ML models.

The models using abstractmodel contains hyperparameter specification, the ML model itself, and training, validation logic.

You can find an example in `models/mnist_cnn.py`

`environment.py` defines a gym environment that training a abstractmodel. It contains state, action, reward logic.

`hyperparameter.py` defines the general hyperparameter space.

### First Experiment

![Figure_1](https://user-images.githubusercontent.com/53331577/144956770-21884871-0c8e-4cfa-9b28-8ed7c88ffa14.png)

### Experiment Result

#### Average reward of workers during epochs

Experiment Condition : 4 workers, 16 steps per episode, 1 dropping, gamma 0.9, update period 4

See [settings.py](settings.py)

![Figure_2](images/average_reward.png)

#### Validation Loss Comparison

Validation loss before training.

![Figure_3](images/val_before.png)

Validation loss after 364 episodes.

![Figure_4](images/val_after.png)

#### Validation Accuracy Comparison

Without A3CHyperOptim : Using Adam optimizer with lr=0.001, weight_decay=0

With A3CHyperOptim : Using Adam with A3CHyperOptim with dynamic lr and weight_decay ((0, 0.002), (1e-6, 1e-4))

![Figure_5](images/val_acc.png)