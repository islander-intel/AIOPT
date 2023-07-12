import pandas as pd
'''https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html'''

def _overfitingCheck(test_acc,train_acc,threshold = 0.2):
    # data = pd.read_csv(f"{filename}.csv")
    # indexthreshold = [index for index,val in enumerate(data["train accuracy"]) if val >=threshold]
    if test_acc>train_acc and (test_acc-train_acc)>threshold:
        return True
    return False
def _underfitingCheck(test_acc,train_acc,threshold = 0.2):
    if train_acc>test_acc and (train_acc-test_acc)>threshold:
        return True
    return False