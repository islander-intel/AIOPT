from collections  import OrderedDict
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from lenet import LeNet
from master import run
params = OrderedDict(
    lr = [.01, .001],
    batch_size = [64,100, 1000],
    shuffle = [True, False],
)
epochs = 30
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
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