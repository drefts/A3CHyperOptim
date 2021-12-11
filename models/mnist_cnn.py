import time
import torch
import re
from torch._C import device
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from abstractmodel import AbstractModel
from hyperparameter import Hyperparameter

from logger import Logger

from settings import GPU_MAP

devicetype = 'cuda' if torch.cuda.is_available() else 'cpu'

devicecount = torch.cuda.device_count()

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

data_loader = torch.utils.data.DataLoader(dataset=training_data ,
                                                batch_size=48,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=0)

class MNIST_CNN(AbstractModel):
    def __init__(self, hyper: Hyperparameter, name: str = '') -> None:
        super().__init__(hyper, name=name)
        self.model = None

    @staticmethod
    def HyperparameterSpecification(): # Hyperparameter Specification
        # Specify your hyperparameters
        hyperparameter = Hyperparameter()
        hyperparameter.Register("LearningRate", 0.001, (0, 0.002), False, False) # Float, Changable
        hyperparameter.Register("WeightDecay", 1e-5, (1e-6, 1e-4), False, False) # Float, Changable

        # Model structure related hyperparameter was disabled
        # hyperparameter.Register("FC_INPUT_SIZE", 5, (5, 20), True, True) # Int, Determined on startup
        return hyperparameter
        
    def Build(self): # Build Model
        _superself = self
        # Define Your Model Here

        # Get CUDA Information
        device_id = re.search(r'\d+', self.name)
        if device_id is None:
            self.device = devicetype
            Logger.Print(self.name, False, "Allocated on fallback device", self.device)
        else:
            self.device = torch.device("cuda", GPU_MAP[int(device_id.group())])
            # self.device = torch.device("cuda", int(device_id.group()) % devicecount)
            Logger.Print(self.name, True, "Allocated on device", self.device)

        # MNIST CNN MODEL
        class CNN(torch.nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.keep_prob = 0.5
                # L1 ImgIn shape=(?, 28, 28, 1)
                #    Conv     -> (?, 28, 28, 32)
                #    Pool     -> (?, 14, 14, 32)
                self.cm_layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2))
                # L2 ImgIn shape=(?, 14, 14, 32)
                #    Conv      ->(?, 14, 14, 64)
                #    Pool      ->(?, 7, 7, 64)
                self.cm_layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2))
                # L3 ImgIn shape=(?, 7, 7, 64)
                #    Conv      ->(?, 7, 7, 128)
                #    Pool      ->(?, 4, 4, 128)
                self.cm_layer3 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

                # L4 FC 4x4x128 inputs -> 625 outputs
                self.fc_layer1 = torch.nn.Linear(4 * 4 * 128, 10, bias=True)
                torch.nn.init.xavier_uniform_(self.fc_layer1.weight)

            def forward(self, x):
                out = self.cm_layer1(x)
                out = self.cm_layer2(out)
                out = self.cm_layer3(out)
                out = out.view(out.size(0), -1)   # Flatten them for FC
                out = self.fc_layer1(out)
                return out
        # END OF MODEL

        # OPTIMIZER
        class AdjustableLearningRateOptimizer:
            def __init__(self, params):
                self.params = params
                
            def zero_grad(self):
                for p in self.params:
                    p.grad = None
            
            def step(self):
                lr = _superself.hyper.Get("LearningRate")
                with torch.no_grad():
                    for p in self.params:
                        p -= lr * p.grad
        # END OF OPTIMIZER

        self.model = CNN().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.hyper.Get("LearningRate"), 
            weight_decay=self.hyper.Get("WeightDecay")
        )
    
    def Train(self) -> float: # Train Model (for each Step)
        
        batch_time = time.time()
        
        avg_cost = 0
        total_batch = len(data_loader)

        for i in range(len(self.optimizer.param_groups)):
            # Apply LR
            self.optimizer.param_groups[i]['lr'] = self.hyper.Get("LearningRate")
            # Apply WeightDecay        
            self.optimizer.param_groups[i]['weight_decay'] = self.hyper.Get("WeightDecay")

        for X, Y in data_loader: # mini batch - label
            X = X.to(self.device)
            Y = Y.to(self.device)
            
            self.optimizer.zero_grad()
            hypothesis = self.model(X)
            cost = self.criterion(hypothesis, Y)
            cost.backward()
            avg_cost += cost / total_batch
            self.optimizer.step()
        
        batch_time = time.time() - batch_time
        
        Logger.Print(self.name, True, '[Elapsed : {}] cost = {:>.9}'.format(batch_time, avg_cost))
        
        return avg_cost

    
    def Validate(self) -> tuple: # Validate Model
        with torch.no_grad():
            X_test = test_data.test_data.view(len(test_data), 1, 28, 28).float().to(self.device)
            Y_test = test_data.test_labels.to(self.device)

            prediction = self.model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            loss = self.criterion(prediction, Y_test)
        return loss, accuracy
    
    def Predict(self): # Make Prediction
        pass

