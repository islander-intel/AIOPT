from collections  import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter # TensorBoard support

import torchvision
# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
from IPython.display import clear_output
from .clear_terminal import clear
# from .checkingfit import overfitingCheck,underfitingCheck
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 

class ModelBuilder():
  def __init__(self):

    # tracking every epoch count, loss, accuracy, time
    self.epoch_count = 0
    self.train_epoch_loss = 0
    self.epoch_train_num_correct = 0

    # tracking every run count, run data, hyper-params used, time
    self.count = 0
    self.run_data = []
    # self.percent_count = 0

  # record the count, hyper-param, model, loader of each run
  # record sample images and network graph to TensorBoard  
  def begin_run(self, run, model, train_data,test_data):

    self.start_time = time.time()

    self.params = run
    self.count += 1

    self.model = model
    self.train_dataloader = train_data
    self.test_dataloader = test_data
    # self.tb = SummaryWriter(comment=f'-{run}')

    train_X,train_y = next(iter(self.train_dataloader))
    test_X,test_y = next(iter(self.test_dataloader))
    grid = torchvision.utils.make_grid(train_X)

    # self.tb.add_image('images', grid)
    # self.tb.add_graph(self.model, train_X)
    # self.tb.add_graph(self.model,test_X)

  # when run ends, close TensorBoard, zero epoch count
  def end_run(self):
    # self.tb.close()
    self.epoch_count = 0

  # zero epoch count, loss, accuracy, 
  def begin_epoch(self):
    self.epoch_start_time = time.time()

    self.epoch_count += 1
    self.train_epoch_loss = 0
    self.epoch_train_num_correct = 0

  # 
  def end_epoch(self):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.start_time

    # record epoch loss and accuracy
    train_loss = self.train_epoch_loss / len(self.train_dataloader.dataset)
    train_accuracy = self.epoch_train_num_correct / len(self.train_dataloader.dataset)

    # Record epoch loss and accuracy to TensorBoard 
    # self.tb.add_scalar('Loss', train_loss, self.epoch_count)
    # self.tb.add_scalar('Accuracy', train_accuracy, self.epoch_count)
# TODO needed to figure out how to get thos WORKING
    # # Record params to TensorBoard 
    # for name, param in self.model.named_parameters():
    #   self.tb.add_histogram(name, param, self.epoch_count)
    #   # print()
    #   self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
    
    # Write into 'results' (OrderedDict) for all run related data
    results = OrderedDict()
    results["run"] = self.count
    results["epoch"] = self.epoch_count
    results["train loss"] = train_loss
    results["train accuracy"] = train_accuracy
    results["test loss"] = self.test_epoch_loss
    results["test accuracy"] = self.test_correct
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration

    

    # Record hyper-params into 'results'
    for k,v in self.params._asdict().items(): results[k] = v
    self.run_data.append(results)
    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    # display epoch information and show progress
    clear_output(wait=True)
    clear()
    # self.percent_count+=1
    # display(df)
    # print((self.percent_count/len(self.run_data)))
    print(df.sort_values(by=["train accuracy","test accuracy"], ascending=True))

  # accumulate loss of batch into entire epoch loss
  def track_test_loss(self,loss):
    self.test_epoch_loss = loss
  def track_test_num_correct(self,correct):
    self.test_correct = correct
  def track_train_loss(self, loss):
    # multiply batch size so variety of batch sizes can be compared
    self.train_epoch_loss += loss * self.train_dataloader.batch_size

  # accumulate number of corrects of batch into entire epoch num_correct
  def track_train_num_correct(self, preds, labels):
    self.epoch_train_num_correct += self._get_num_correct(preds, labels)

  @torch.no_grad()
  def _get_num_correct(self, preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
  
  # save end results of all runs into csv, json for further analysis
  


  def save(self, fileName):

    data = pd.DataFrame.from_dict(
        self.run_data, 
        orient = 'columns',
    ).sort_values(by=["train accuracy","test accuracy"], ascending=True).to_csv(f'{fileName}.csv',index=False)


    with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
      json.dump(self.run_data, f, ensure_ascii=False, indent=4)
