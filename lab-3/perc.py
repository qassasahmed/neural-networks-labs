import numpy as np


def activation_function(y_in):
    """
    Threshold Activation function with Theta equals zero.
    :param y_in: the linear summation of wi * xi
    :return:  0 if y_in == 0, 1 if y_in > 0, -1 if y_in <0
    """
    print(f"y_in:: {y_in : <10}  y_out:: {0 if y_in == 0 else y_in / abs(y_in)}")
    return 0 if y_in == 0 else y_in / abs(y_in)


def y_o(inputs, weights):
    """
    Calculates the linear summation of wi * xi, then calls the activation function
    :param inputs: list of xi
    :param weights: list of wi
    :return: y_o which the non-linear threshold activation of the linear summation
    """
    summation = 0
    for i in range(len(inputs)):
        summation += inputs[i] * weights[i]
    return activation_function(summation)


def update_weights(w, d):
    """

    :param w:
    :param d:
    :return:
    """
    return [w[i] + d[i] for i in range(len(w))]


class Perceptron:

    def __init__(self, number_of_features, eta):
        self.old_weights = list(np.zeros(number_of_features))
        self.no_delta_update = list(np.zeros(number_of_features))
        self.deltas = []
        self.eta = eta
        self.current_epoch = []
        self.all_epochs = []

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
                y_output = y_o(row, self.old_weights)
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


if __name__ == '__main__':
    print("go to main")
