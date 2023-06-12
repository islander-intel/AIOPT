
from torchvision import transforms
from torchvision import datasets
from lenet import LeNet
# from AiOpt import master
from src.master import torch_run
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
    torch_run(LeNet,params,train_set,test_set,epochs)

if __name__ == "__main__":
    main()