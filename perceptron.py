import matplotlib.pyplot as plt
from time import process_time
import numpy as np



class perceptron:

    def __init__(self,inputs, learning_rate, neuron, bias, epochs):
        self.data = inputs
        self.l_r = learning_rate
        self.i_p = neuron
        self.bias = bias
        self.epochs = epochs
        self.w = np.zeros([self.i_p, 1])
        self.w = np.insert(self.w, 0, bias)
        self.w1 = np.transpose(self.w)
        self.err = 0

    def train(self,n_tr):
        st = process_time()
        ee = np.zeros(n_tr)
        mse = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            shuff_seq = np.random.permutation(n_tr)
            data_shuffle_tr = self.data[:, shuff_seq]

            for i in range(n_tr):
                x = data_shuffle_tr[0:2, i]
                x = np.insert(x, 0, 1)
                d = data_shuffle_tr[2, i]
                y = np.sign(np.dot(self.w1, x))
                ee[i] = d - y
                w_n = self.w1 + np.dot((self.l_r * (d - y)), x)
                self.w1 = w_n
            mse[epoch] = (np.square(ee)).mean(axis=0)
        st2 = process_time()

        print('No.of Training data: %d\n' % n_tr)
        print('time taken :%f seconds\n' % (st - st2))
        plt.title("Training accuracy Curve")
        plt.xlabel("epochs")
        plt.ylabel("mse")
        plt.plot(mse)
        plt.show()

    def test(self, n_te, n_tr):

        shuff_seq = np.random.permutation(n_tr)
        data_shuffle_te = self.data[:, shuff_seq]
        for i in range(n_te + 1):
            x = data_shuffle_te[0:2, i]
            x = np.insert(x, 0, 1)
            d = data_shuffle_te[2, i]
            y = np.sign(np.dot(self.w1, x))
            if y == 1:
                plt.plot(x[1], x[2], 'rx')
            elif y == -1:
                plt.plot(x[1], x[2], 'k+')
            elif not (d - y) == 0:
                self.err += 1

        print('Point Tested : {0}\n'.format(n_te))
        print("Error Points : {0} ({1}%)\n".format(self.err, ((self.err / n_te) * 100)))

        plt.title("Classification using Perceptron")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

