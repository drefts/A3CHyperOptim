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

![Figure_2](images/average_reward.png)

![Figure_3](images/val_before.png)

![Figure_4](images/val_after.png)