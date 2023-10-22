import pandas as pd

truth_table = pd.read_csv("datasets/xor-gate.csv")
print(truth_table)

x1 = truth_table["x1"]
x2 = truth_table["x2"]
bias = truth_table["b"]
T = truth_table["T"]
alpha = 1

delta_x1 = []
delta_x2 = []
delta_b = []


def delta_values():
    for i in range(len(x1)):
        delta_x1.append(alpha * x1[i] * T[i])
    for i in range(len(x2)):
        delta_x2.append(alpha * x2[i] * T[i])
    for i in range(len(bias)):
        delta_b.append(alpha * bias[i] * T[i])


# deltas = pd.DataFrame(list(zip(delta_x1, delta_x2, delta_b)), columns=["d_x1", "d_x2", "d_bias"])
# print(deltas)
def update_values(old_wights):
    """
    w new = w old + delta
    :return: w1 w2 b
    """
    new_x1 = [old_wights[0]]
    new_x2 = [old_wights[1]]
    new_b = [old_wights[2]]
    delta_values()
    for i in range(len(x1)):
        new_x1.append(delta_x1[i] + new_x1[-1])
    for i in range(len(x2)):
        new_x2.append(delta_x2[i] + new_x2[-1])
    for i in range(len(bias)):
        new_b.append(delta_b[i] + new_b[-1])

    new_weights = pd.DataFrame(list(zip(new_x1, new_x2, new_b)))
    print(new_weights)
    return new_x1[-1], new_x2[-1], new_b[-1]


def heb_net():
    w1, w2, b = update_values([0, 0, 0])
    for i in range(5):
        Y = []
        for i in range(len(x1)):
            Y.append(-1 if w1 * x1[i] + w2 * x2[i] + b < 0 else 1)

        if Y == list(T):
            print(Y)
            break
        else:
            w1, w2, b = update_values([w1, w2, b])


heb_net()




