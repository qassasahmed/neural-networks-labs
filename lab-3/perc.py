import numpy as np


def activation_function(y_in):
    return 0 if y_in == 0 else y_in / abs(y_in)


def y_o(inputs, weights):
    """
    y_in = sum(wi * xi)
    :return: activation_function(sum)
    """
    sum = 0
    for i in range(len(inputs)):
        sum += inputs[i] * weights[i]
    return activation_function(sum)


def update_weights(w, d):
    """
    w new = w old + delta
    :return: list of new weights
    """
    return [w[i] + d[i] for i in range(len(w))]


class Perceptron:

    def __init__(self, number_of_features):
        self.old_weights = list(np.zeros(number_of_features))
        self.no_update = list(np.zeros(number_of_features))
        self.eta = 1
        self.deltas = []

    def run(self, matrix, target):
        """
        1] y_o(row, initial weights)
        2] check y_o == target ->>>> flag
        3] delta_values()
        4] new weights
        5] old_wights = new weights
        6] repeat
        :return: epoch weights
        """
        i = 0
        for row in matrix:
            d = self.delta_values(y_o(row, self.old_weights, ) == target[i], row, target[i])
            self.old_weights = update_weights(self.old_weights, d)
            i = i + 1
        return self.old_weights

    def delta_values(self, flag, row, T):
        """
        delta = xi * eta * T if x y_o == T else 0
        :return: list of deltas
        """
        if flag:
            return self.no_update
        else:
            self.deltas = [i * self.eta * T for i in row]
            return self.deltas
