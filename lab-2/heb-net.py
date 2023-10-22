import pandas as pd

truth_table = pd.read_csv("datasets/and-gate.csv")

x1 = truth_table["x1"]
x2 = truth_table["x2"]
bias = truth_table["b"]
T = truth_table["T"]
alpha = 1
delta_x1 = []
delta_x2 = []
delta_bias = []
print(x1)


def delta_values():
    for i in range(len(x1)):
        delta_x1.append(alpha * x1[i] * T[i])
        delta_x2.append(alpha * x2[i] * T[i])
        delta_bias.append(alpha * bias[i] * T[i])


# delta_values()
# deltas = pd.DataFrame(list(zip(delta_x1, delta_x2, delta_bias)))
# print(deltas)


def update_weights(old_weights):
    new_x1 = [old_weights[0]]
    new_x2 = [old_weights[1]]
    new_bias = [old_weights[2]]
    delta_values()
    for i in range(len(x1)):
        new_x1.append(new_x1[i] + delta_x1[i])
    for i in range(len(x2)):
        new_x2.append(new_x2[i] + delta_x2[i])
    for i in range(len(bias)):
        new_bias.append(new_bias[i] + delta_bias[i])

    new_weights = pd.DataFrame(list(zip(new_x1, new_x2, new_bias, delta_x1, delta_x2, delta_bias)),
                               columns=["w1", "w2", "bias", "d_x1", "d_x2", "d_bias"])
    print(new_weights)
    return new_x1[-1], new_x2[-1], new_bias[-1]


def hebbian():
    w1, w2, b = update_weights([0, 0, 0])
    for _ in range(5):
        y = []
        for i in range(len(x1)):
            y.append(-1 if w1 * x1[i] + w2 * x2[i] + b * bias[i] < 0 else 1)
        print(f"Output: {y}")
        if y == list(T):
            break
        else:
            w1, w2, b = update_weights([w1, w2, b])


hebbian()
