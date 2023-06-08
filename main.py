from collections  import OrderedDict
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from lenet import LeNet
from master import run
# from collections  import UserDict
# params = OrderedDict(
#     lr = [.01, .001],
#     batch_size = [64,100, 1000],
#     shuffle = [True, False],
# )
params = {
    "lr" : [.01, .001],
    "batch_size": [64,100, 1000],
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
    run(LeNet,params,train_set,epochs)

if __name__ == "__main__":
    main()