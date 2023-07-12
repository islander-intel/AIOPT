import pandas as pd

def best(filename,threshold=0.8):
    data = pd.read_csv(f"{filename}.csv")
    best_traning = [index for index,val in enumerate(data["train accuracy"]) if val >=threshold]
    best_test = [index for index,val in enumerate(data["test accuracy"]) if val >=threshold]
    best_combined = list(set(best_traning+best_test)).sort()
    columns = data.columns()
    temp = []
    for i in best_combined:
        temp.append(data.iloc[i,:])
    pd.DataFrame(temp,columns=columns).to_csv(f"{filename}_best.csv")