import pandas as pd
import percv2


def read_file(path):
    df = pd.read_csv(path)
    target = df.iloc[:, -2:]
    features = df.iloc[:, :-1].values.tolist()
    return features, target


x, T = read_file("datasets/data.csv")
p1 = percv2.Perceptron(len(x[0]), 1)
p1.run(x, T["T1"].values.tolist())

p2 = percv2.Perceptron(len(x[0]), 1)
p2.run(x, T["T2"].values.tolist())
