# put all hyper params into a OrderedDict, easily expandable
#  import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
import torch
import torch.nn.functional as F
import torch.optim as optim
# import torchvision module to handle image manipulation
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 
from modelbuilder import ModelBuilder
from modelcombination import ModelCombination
def run(NNmodel,params,data,epochs):
    m = ModelBuilder()
    # count = 0
    # get all runs from params using RunBuilder class
    for run in ModelCombination.get_runs(params):

        # if params changes, following line of code should reflect the changes too
        model = NNmodel()
        loader = torch.utils.data.DataLoader(data, batch_size = run.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        m.begin_run(run, model, loader)
        for epoch in range(epochs):
            m.begin_epoch()
            for batch in loader:
                X = batch[0]
                y = batch[1]
                preds = model(X)
                loss = F.cross_entropy(preds,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds, y)

            m.end_epoch()
            # count+=1
            # percent = len()
        m.end_run()

    # when all runs are done, save results to files
    m.save('results')