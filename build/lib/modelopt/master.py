
from collections  import OrderedDict
import torch
import torch.nn.functional as F
import torch.optim as optim
from .build import train,test
# import torchvision module to handle image manipulation
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 
from .modelbuilder import ModelBuilder
from .modelcombination import ModelCombination
# import torch
#TODO need to add an override aspect to device
def torch_run(NNmodel,params,test_data,train_data,epochs,device = None):
    if device == None:
        device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"Using {device} device")
    torch.cuda.empty_cache()
    # params = OrderedDict(params)
    params = OrderedDict(
        lr = params["lr"],
        batch_size = params["batch_size"],
        shuffle = params["shuffle"],
    )
    m = ModelBuilder()
    # count = 0
    # get all runs from params using RunBuilder class
    for run in ModelCombination.get_runs(params):

        # if params changes, following line of code should reflect the changes too
        model = NNmodel()
        model.to(device)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = run.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=run.lr)# this part is what I want to be able to opt
        test_loader = torch.utils.data.DataLoader(test_data,batch_size = run.batch_size)
        m.begin_run(run, model, train_loader,test_loader)
        for epoch in range(epochs):
            m.begin_epoch()
            loss_fn = F.cross_entropy
            train(train_loader,model,loss_fn,optimizer,device,m)
            test(test_loader,model,loss_fn,device,m)
            m.end_epoch()

        m.end_run()

    # when all runs are done, save results to files
    m.save('results')

if __name__=="__main__":
    from torchvision import transforms
    from torchvision import datasets
    from lenet import LeNet
    from master import torch_run
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
    torch_run(LeNet,params,train_set,test_set,epochs)
