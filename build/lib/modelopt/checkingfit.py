import pandas as pd
'''https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html'''

def overfitingCheck(test_loss,train_loss):
    # data = pd.read_csv(f"{filename}.csv")
    # # indexthreshold = [index for index,val in enumerate(data["train accuracy"]) if val >=threshold]
    # if test_loss>train_loss:
    #     return True
    # return False
    return test_loss>train_loss
def underfitingCheck(test_acc,train_acc,threshold = 0.2):
    # Overfitting: train loss continues to decrease while test/val loss increases
# Underfitting: train loss remains high and doesn’t decrease(not constant). can be better determined using accuracy on train set rather than loss.
# Best fit: Best to choose the weights giving high accuracy(or any metric depending on the problem) on test/val set rather than finding it with loss.

# In the above graph, you can choose the weights at Early Stopping Checkpoint point.
    if train_acc>test_acc and (train_acc-test_acc)>threshold:
        return True
    return False