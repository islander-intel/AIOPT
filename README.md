# AI OPT

## AI OPT is a Python module for PyTorch developers to be able to find the most optimal parameters for their Classification model. The package is distributed under the 3-Clause BSD license.

AI OPT is a Python package developed to enable PyTorch developers to find the best parameters for their deep-learning classification model. such as convolutional neural network(CNN), and recurring neural network(RNN). Developers are able t to save time in finding the best combination of parameters to ensure an accurate model.

## Dependency

PyTorch

pandas 

NumPy

sci-kit learn

 

## pip install
Comming soon but will be the command bellow
```bash
pip install aiopt
```

## Example of use

```python
from torchvision import transforms
from torchvision import datasets
from aiopt import torch_run_classification as torch_run
# from torch_class import simple_class
from torch import nn

class simple_class(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28,512),
            nn.SELU(),
            nn.Linear(512,10),
            nn.LogSoftmax(dim=1)
        )
        self.flatten = nn.Flatten()

    def forward(self,data):
        X = self.flatten(data)
        return self.model(X)
params = {
    "lr" : [.01, .001,.1],
    "batch_size": [i for i in range(64,100)],
    "shuffle": [True, False],
}
epochs = 3
train_set = datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
test_set = datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
def main():
    torch_run(simple_class,params,train_set,test_set,epochs)

if __name__ == "__main__":
    main()
```

## Video example:

For those of you who are more visual learners, there will be a link to a video on my youtube channel where I go over everything you needed to know as well as how to use AI OPT

## Upcoming improvements/ fixed issues:

if you find any issues you would like to report please see the link below to report any issues. before you do so please check the list below to ensure it is not a known issue.

- Tools to be able to evaluate the results
- TensorFlow model capabilities
- regression model capabilities
- efficiency improvements
