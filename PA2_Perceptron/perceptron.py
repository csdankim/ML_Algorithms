import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import lru_cache, partial
import pickle
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

@lru_cache(maxsize=5)
def load_data(data_type = 'train'):

    if data_type == 'test':
        data_type = 'test_no_label'

    print('Loading {}'.format(data_type))

    data_root = os.path.join(ROOT, 'data')
    file_path = os.path.join(data_root, 'pa2_{}.csv'.format(data_type))
    data = np.genfromtxt(file_path, delimiter=',')

    y = None
    if data_type != 'test_no_label':
        # Transform first col, 3->1 and 5->-1
        y = -(data[:,0]-4)
        data = data[:, 1:]

    # Add bias
    bias_col = np.zeros((data.shape[0], 1)) + 1
    data = np.concatenate([data, bias_col], axis=1)

    print('Done loading {}'.format(data_type))

    return y, data


class Perceptron(object):

    def __init__(self, n):

        if isinstance(n, np.ndarray):
            n = n.shape[1]

        self.w = np.zeros(n)

    def train(self, y, data, iter = 1):

        misclassified = []
        for _ in range(iter):
            for i in range(data.shape[0]):

                x = data[i, :]
                updated = self.update_weights(y[i], x)
                if updated:
                    misclassified.append(i)

        return misclassified

    def update_weights(self, y, x):
        raise NotImplementedError

    def predict(self, data):
        return np.sign(data.dot(self.w)).astype(np.int)

    def validate(self, data, y):
        return (self.predict(data) == y).mean()


class OnlinePerceptron(Perceptron):

    def update_weights(self, y, x):

        misclassified = False
        ind = y * np.dot(self.w, x)
        if ind <= 0:
            self.w += y * x
            misclassified = True

        return misclassified

class AveragePerceptron(Perceptron):

    def __init__(self, n):

        super(AveragePerceptron, self).__init__(n)
        self.w_avg = self.w.copy()
        self.counter = 1

    def update_weights(self, y, x):
        misclassified = False
        ind = y * np.dot(self.w, x)
        if ind <= 0:
            self.w += y * x
            misclassified = True

        self.w_avg = (self.counter * self.w_avg + self.w) / (self.counter + 1)
        self.counter += 1
        return misclassified

    def predict(self, data):
        return np.sign(data.dot(self.w_avg))


class KernelPerceptron(Perceptron):

    def __init__(self, data, y, p):

        self.data = data
        self.y = y

        self.p = p
        self.K = None
        self.a = np.zeros(data.shape[0])
        self.load_gram_matrix()

    def train(self, y, data, iter=1):

        for _ in range(iter):
            for i in range(data.shape[0]):
                self.update_weights(y, i)

    def update_weights(self, y_vec, x_ind):

        u = (self.a * self.K[x_ind, :] * y_vec).sum()
        if y_vec[x_ind] * u <= 0:
            self.a[x_ind] += 1

    def load_gram_matrix(self):

        n = self.data.shape[0]
        self.K = np.zeros((n, n))

        self.K = self.data.dot(self.data.T)
        if isinstance(self.p, int):
            self.K = np.power(1 + self.K, self.p)

    def predict(self, data):

        gram = data.dot(self.data.T)
        if isinstance(self.p, int):
            gram = np.power(1 + gram, self.p)

        return np.sign((gram * (self.a * self.y)).sum(axis=1)).astype(np.int)



# Utility functions
def evaluate_accuracy(perceptron, iters=15):

    y_training, data_training = load_data('train')
    y_validation, data_validation = load_data('valid')

    iters = np.arange(0, iters) + 1

    rez = pd.DataFrame(columns=['Training', 'Validation'], index=iters)
    for iter in iters:
        perceptron.train(y_training, data_training, 1)
        rez.loc[iter, 'Training'] = perceptron.validate(data_training, y_training)
        rez.loc[iter, 'Validation'] = perceptron.validate(data_validation, y_validation)

    return rez


if __name__ == '__main__':

    y_training, data_training = load_data('train')
    _, data_test = load_data('test_no_label')

    mode = 'default'
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == 'validate':
        # This section is for testing to make sure the kernel perceptron and the main perceptron line up!
        online = OnlinePerceptron(data_training)
        k_equiv = KernelPerceptron(data_training, y_training, 'default')

        for _ in range(15):
            online.train(y_training, data_training)
            k_equiv.train(y_training, data_training)

        abs_diff = np.abs(online.w - k_equiv.w).sum()
        print('Abs diff between final weights was {}'.format(abs_diff))
        sys.exit(0)

    # By default, loads all perceptrons and plots their accuracies after 15 iterations

    if mode in ['online', 'default']:
        print('Evaluating online perceptron...')
        online = OnlinePerceptron(data_training)
        rez = evaluate_accuracy(online)
        best_iters = rez['Validation'].argmax()
        print('Online best iters value was:', best_iters)

        best_online = OnlinePerceptron(data_training)
        best_online.train(y_training, data_training, iter=best_iters)
        np.savetxt('oplabel.csv', best_online.predict(data_test), fmt='%i')

        rez.plot()
        plt.title('Online Perceptron')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('online_perceptron.png')
        plt.clf()


    if mode in ['average', 'default']:
        print('Evaluating average perceptron...')
        average = AveragePerceptron(data_training)
        rez = evaluate_accuracy(average)
        best_iters = rez['Validation'].argmax()
        print('Average best iters value was:', best_iters)

        best_average = AveragePerceptron(data_training)
        best_average.train(y_training, data_training, iter=best_iters)
        np.savetxt('aplabel.csv', best_average.predict(data_test), fmt='%i')

        rez.plot()
        plt.title('Average Perceptron')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('average_perceptron.png')
        plt.clf()

    if mode in ['kernel', 'default']:
        if len(sys.argv) > 2:
            ps = [int(sys.argv[2])]
        else:
            ps = [1, 2, 3, 4, 5]

        all_rez = []

        for p in ps:
            print('Evaluating kernel perceptron (p={})'.format(p))
            kernel = KernelPerceptron(data_training, y_training, p)

            rez = evaluate_accuracy(kernel)
            rez['p'] = p
            all_rez.append(rez.set_index([rez.index, 'p']))

        all_rez = pd.concat(all_rez)
        best_i, best_p = all_rez['Validation'].argmax()

        print('Best kernel perceptron model was p={} after {} iterations'.format(best_p, best_i))

        best_kernel = KernelPerceptron(data_training, y_training, best_p)
        best_kernel.train(y_training, data_training, best_i)
        np.savetxt('kplabel.csv', best_kernel.predict(data_test), fmt='%i')

        # Here we plot training and validation accuracies separately
        fig_tr, fig_val = plt.figure(), plt.figure()
        ax_tr = fig_tr.add_subplot(111)
        ax_val = fig_val.add_subplot(111)

        p_best_val = {}
        p_best_train = {}
        for p, df_subset in all_rez.reset_index('p').groupby('p'):
            ax_tr.plot(df_subset['Training'], label='p={}'.format(p))
            ax_val.plot(df_subset['Validation'], label='p={}'.format(p))
            p_best_val[p] = df_subset['Validation'].max()
            p_best_train[p] = df_subset['Training'].max()

        p_best_train = pd.Series(p_best_train).sort_index()
        p_best_val = pd.Series(p_best_val).sort_index()

        plt.clf()
        plt.plot(p_best_train)
        plt.plot(p_best_val)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('p')
        plt.ylabel('Accuracy')
        plt.title('Kernel Perceptron Accuracy vs. Power')
        plt.savefig('kernel_accuracy.png')


        ax_tr.set_title('Kernel Perceptron Training Accuracy')
        ax_tr.set_xlabel('Iterations')
        ax_tr.set_ylabel('Accuracy')
        ax_tr.legend()

        ax_val.set_title('Kernel Perceptron Validation Accuracy')
        ax_val.set_xlabel('Iterations')
        ax_val.set_ylabel('Accuracy')
        ax_val.legend()

        fig_tr.savefig('kernel_training.png')
        fig_val.savefig('kernel_validation.png')
