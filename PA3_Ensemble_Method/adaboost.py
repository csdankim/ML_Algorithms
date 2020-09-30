import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from decision_tree import BinaryDecisionTree, load_data

class Adaboost(object):

    def __init__(self, max_depth = 1):
        self.max_depth = max_depth
        self.alphas = []
        self.estimators = []
        self.error_weights = pd.Series(1.0, index=data_class.index) / len(data_class)

    def train(self, data, data_class, n=1):
        """
        Note that for training, you can keep training the model after it's already been trained because it
        automatically remembers the last error weights.
        So self.train(n=2) is equivalent to self.train(n=1); self.train(n=1)
        """
        # Initialize the sample weights

        for _ in range(n):

            # Train the weak estimator
            new_estimator = BinaryDecisionTree(self.max_depth)
            new_estimator.train(data, data_class, self.error_weights)
            y_pred, _ = new_estimator.validate(data, data_class)

            # Check misclassifications and update the error weights accordingly
            incorrect = y_pred != data_class
            error = self.error_weights[incorrect].sum()
            alpha = 0.5 * np.log((1.0 - error) / error)
            sign = incorrect * 2 - 1        # Maps incorrect to 1, correct to -1
            self.error_weights *= np.e ** (sign * alpha)
            self.error_weights /= self.error_weights.sum()

            # Save down alpha and estimator for prediction
            self.alphas.append(alpha)
            self.estimators.append(new_estimator)

    def classify(self, data):

        classifications = pd.Series(0, index = data.index)
        weighted_predictions = [alpha * (estimator.classify(data) * 2 - 1) for alpha, estimator in zip(self.alphas, self.estimators)]
        # Need to map individual predictions to -1 and 1

        aggregate_predictions = reduce(lambda x, y: x+y, weighted_predictions)
        classifications[aggregate_predictions > 0] = 1

        return classifications

    def validate(self, data, data_class):
        classifications = self.classify(data)
        accuracy = (data_class == classifications).mean()
        return classifications, accuracy

if __name__ == '__main__':
    data, data_class = load_data('train')
    val, val_class = load_data('val')

    # 3_c #
    print("--> Start Adaboost c.")
    print()
    ada = Adaboost(1)
    L = [1, 2, 5, 10, 15]
    train_acc = []
    val_acc = []

    for l in L:
        print("L is {}".format(l))

        ada.train(data, data_class, l)
        _, accuracy = ada.validate(data, data_class)
        _, val_accuracy = ada.validate(val, val_class)

        print('AdaBoost achieved {:.3f}% accuracy on training data'.format(accuracy * 100))
        print('AdaBoost achieved {:.3f}% accuracy on validation data'.format(val_accuracy * 100))
        print()
        train_acc.append(accuracy)
        val_acc.append(val_accuracy)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(L, train_acc, label="train")
    ax1.plot(L, val_acc, label="validate")
    ax1.legend()
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('L')
    plt.title('Adaboost by L')
    plt.show()

    print("--> Finished Adaboost c.")
    print()

    # 3_e #
    print("--> Start Adaboost e. with depth = 2, L = 6")
    print()
    ada = Adaboost(2)
    ada.train(data, data_class, 6)
    _, accuracy = ada.validate(data, data_class)
    _, val_accuracy = ada.validate(val, val_class)

    print('AdaBoost achieved {:.3f}% accuracy on training data'.format(accuracy * 100))
    print('AdaBoost achieved {:.3f}% accuracy on validation data'.format(val_accuracy * 100))
    print()
    print("--> Finished Adaboost e.")
    print()

    # 3_f #
    print("--> Start Adaboost f. with depth = 2, L = 6")
    print()

    test, _ = load_data('test')

    ada = Adaboost(1)
    ada.train(data, data_class, 15)
    res, _ = ada.validate(test, _)

    print(res)
    np.savetxt('pa3_test.csv', res, fmt='%i')
    print("--> created pa3_test.csv")
    print()
    print("--> adaboost finished")


