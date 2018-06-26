from sys import argv
from random import shuffle
import numpy as np

class LSGD(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters = 100, batch_size=100):
        """
        Input
        :param X:
        :param y:
        :param learning_rate:
        :param reg:
        :param num_iters:
        :param batch_size:
        :return: A list
        """

        num_train, dim = X.shape
        num_y = y.shape[0]
        print(num_y)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_y)

        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            randomIndex = np.random.choice(len(X), batch_size, replace=True)
            X_batch = X[randomIndex]
            y_batch = y[randomIndex]

            # evaluate loss and gradient
            loss, grad = self.loss(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)
            # Update the weights
            self.W += (-grad*learning_rate)
            if it % 2 == 0:
                print("iteration %d / %d: loss: %f" % (it, num_iters, loss))


    def loss(self, W, X, y, reg):
        dW = np.zeros(W.shape)
        num_y = W.shape[1]
        num_train = X.shape[0]
        print(dW.shape)
        print(num_y)
        print(num_train)
        print(y.shape)
        loss = 0.0
        dW = 0.0
        return loss, dW



