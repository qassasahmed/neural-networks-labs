import numpy as np
import pandas as pd


def read_file(path):
    df = pd.read_csv(path)
    target = df.iloc[:, -1].values.tolist()
    features = df.iloc[:, :-1].values.tolist()
    return features, target


def y_in(inputs, weights):
    """
    Calculates the linear summation of wi * xi
    :param inputs: list of xi
    :param weights: list of wi
    :return: the linear summation of weighted input
    """
    summation = 0
    for i in range(len(inputs)):
        summation += inputs[i] * weights[i]
    return summation


def update_weights(w, d):
    return [w[i] + d[i] for i in range(len(w))]


class Perceptron:

    def __init__(self, number_of_features, eta=1):
        self.old_weights = list(np.zeros(number_of_features))
        self.no_delta_update = list(np.zeros(number_of_features))
        self.deltas = []
        self.eta = eta
        self.current_epoch = []
        self.all_epochs = []
        self.act = Activation()

    def delta_values(self, flag, row, T):
        """
        delta = xi * eta * T if x y_o == T else 0
        :return: list of deltas
        """
        if flag:
            return self.no_delta_update
        else:
            self.deltas = [i * self.eta * T for i in row]
            return self.deltas

    def run(self, matrix, target):
        for epoch in range(5):
            i = 0
            for row in matrix:
                y_input = y_in(row, self.old_weights)
                y_output = self.act.threshold(y_input)
                delta = self.delta_values(y_output == target[i], row, target[i])
                self.old_weights = update_weights(self.old_weights, delta)
                i = i + 1
            print(f"epoch {epoch + 1}: {self.old_weights} \n")
            if self.old_weights == self.current_epoch:
                self.all_epochs.append(self.old_weights)
                print("Right weights were found")
                break
            else:
                # save the weights of current epoch to compare it with next epoch
                self.all_epochs.append(self.old_weights)
                self.current_epoch = self.old_weights

        return self.all_epochs

    def get_activation_info(self):
        return self.act.get_input_output()


class Activation:
    y_in: list[float]
    y_out: list[float]

    def __init__(self):
        self.y_in = []
        self.y_out = []

    def get_input_output(self):
        return pd.DataFrame(list(zip(self.y_in, self.y_out)),
                            columns=["y_in", "y_out"])

    def linear(self, y_input):
        self.y_in.append(y_input)
        self.y_out.append(y_input)
        return self.y_out[-1]

    def threshold(self, y_input):
        self.y_in.append(y_input)
        self.y_out.append(0 if y_input == 0 else y_input / abs(y_input))
        return self.y_out[-1]

    def relu(self, y_input):
        self.y_in.append(y_input)
        self.y_out.append(max(0, y_input))
        return self.y_out[-1]

    def sigmoid(self):
        pass

    def tanh(self):
        pass
