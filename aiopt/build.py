import torch
from torch import nn
from torchvision.transforms import ToTensor
def build(epoch,train_dataloader,test_loader,model,loss_fn,opt,device):
    for e in range(epoch):
        print(f"Epoch {e+1}\n-------------------------------")
        train(train_dataloader,model,loss_fn,opt,device)
        test(test_loader,model,loss_fn,device)
def train(dataloader,model,loss_fn,optimizer,device,m):
    # size = len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        X = X.float()
        pred = model(X)
        loss = loss_fn(pred,y)


        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100==0:
            loss = loss.item()
            m.track_train_loss(loss)
            m.track_train_num_correct(pred, y)



def test(dataloader,model,loss_fn,device,m):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss,correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            X = X.float()
            pred = model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=num_batch
    correct/=size
    m.track_test_loss(test_loss)
    m.track_test_num_correct(correct)