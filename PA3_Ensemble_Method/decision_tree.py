import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from functools import lru_cache, partial
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

class BinaryDecisionTree(object):

    def __init__(self, max_depth=0):
        self.max_depth = max_depth      # When the tree is built, how deep will it be? (Terminal node is max_depth=0)
        self.selector = None            # What data feature are we splitting on?
        self.children = None            # Children trees
        self.node_return = None         # Used for leaves - What return val does this node represent?

    def train(self, data, data_class, data_weights=None):
        """
        Constructs the decision tree based off the input data.
        :param data: A DataFrame with the selectors in the columns.
        :param data_class: A Series indicating the ground truth class.
        :param data_weights: If not None, a Series with the weights of the data (for use in AdaBoost)
        :return: None
        """

        if data_weights is None:
            data_weights = pd.Series(1.0, index=data_class.index) / len(data_class)
        else:
            data_weights /= data_weights.sum()
        # If we're at a terminal node, we return the most likely classification based off the data class
        if not self.max_depth:
            self.node_return = 1 if data_weights[data_class == 1].sum() > 0.5 else 0
            return

        # If we've reached a node in which all of our classifiers are the same, then we're done and we simply
        # return that classifier
        if (data_class == data_class.values[0]).all():
            self.node_return = data_class.values[0]
            return

        # Otherwise, loop through each of the columns looking for the best gini coefficient
        selectors = data.columns

        current_best = None
        current_gini = np.inf

        for selector in selectors:
            idx = data[selector] == 1
            # We cannot classify based off a feature which is all the same
            if idx.all() or (~idx).all():
                continue

            # Compute gini coefficients
            p = (idx * data_weights).sum()
            gini_pos = compute_gini_index(data_class[idx], wgt=data_weights[idx], rescale=True)
            gini_neg = compute_gini_index(data_class[~idx], wgt=data_weights[~idx], rescale=True)
            new_gini = p * gini_pos + (1-p) * gini_neg

            if new_gini < current_gini:
                current_best = selector
                current_gini = new_gini

        # This clause is hit when none of the randomly selected features for a tree in a 
        # random forest are useful.
        if current_best is None:
            #raise Exception('This clause should never be hit unless all of your classifiers are uniform!')
            current_best = data.columns[0]

        self.selector = current_best

        self.children = {
            1: BinaryDecisionTree(max_depth=self.max_depth-1),
            0: BinaryDecisionTree(max_depth=self.max_depth-1)
        }

        idx = data[self.selector] == 1
        # data = data[data.columns.delete(self.selector)]       
        self.children[1].train(data[idx], data_class[idx], data_weights[idx])
        self.children[0].train(data[~idx], data_class[~idx], data_weights[~idx])

    def classify_single(self, example):
        if self.node_return is not None:
            return self.node_return
        return self.children[example[self.selector]].classify_single(example)

    def classify(self, data):

        if isinstance(data, pd.Series):
            return self.classify_single(data)

        return data.apply(self.classify_single, axis=1)


    def validate(self, data, data_class):
        classifications = self.classify(data)
        accuracy = (data_class == classifications).mean()
        return classifications, accuracy

    def visualize(self, _t = 0):
        tabs = '\t' * _t
        if self.node_return is not None:
            print('{}{}'.format(tabs, self.node_return))
            return

        print('{}{}?'.format('\t' * _t, self.selector))
        self.children[1].visualize(_t + 1)
        self.children[0].visualize(_t + 1)


class RandomForest(object):
        def __init__(self, n, m, d, useSeed=False):
            self.n = n # Trees in the forest
            self.m = m # Features per tree
            self.d = d # max depth
            self.forest = []
            self.useSeed = useSeed
            if m<d:
                self.d=m

        def train(self, data, data_class):
            if self.useSeed:
                seed = random.randint(0, 1000)
                np.random.seed(seed=seed)
                print('%s%d' %("Random seed: ", seed))
            else:
                seed = None
            for i in range(self.n):
                #print("Training tree %s" % i)
                tree = BinaryDecisionTree(self.d)
                # Choose data samples
                dataIndicies = np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=True)

                # Choose features
                features = np.random.choice(data.columns, size=self.m, replace=False)

                # Create training data for tree
                featureData = data[features].copy()
                classArray = np.zeros(data.shape[0])
                treeClass = pd.Series([data_class.iloc[i] for i in dataIndicies])
                treeData = pd.DataFrame(columns=features)
                for i in dataIndicies:
                    treeData = treeData.append(featureData.iloc[i], ignore_index=True)

                # Train Tree
                tree.train(treeData, treeClass)
                self.forest.append(tree)


        def classify(self, data):
            classifications = np.zeros(data.shape[0])
            for tree in self.forest:
                classifications = np.add(classifications, tree.classify(data))
            for i in range(data.shape[0]):
                if classifications[i]/float(self.n) > 0.5:
                    classifications[i] = 1
                else:
                    classifications[i] = 0
            return classifications

        def validate(self, data, data_class):
            classifications = self.classify(data)
            accuracy = (data_class == classifications).mean()
            return classifications, accuracy


def compute_gini_index(class_subset, wgt=None, rescale=True):
    """
    Computes the gini index based off of a Series of 0s and 1s.
    :param class_subset: A Series with 0s and 1s (or Trues and Falses)
    :return: A float between 0 and 0.5
    """

    if wgt is None:
        wgt = pd.Series(1.0, index=class_subset.index) / len(class_subset)
    elif rescale:
        wgt /= wgt.sum()

    p = (class_subset * wgt).sum()
    return 1 - p**2 - (1-p)**2


#@lru_cache(maxsize=5)
def load_data(data_type = 'train'):

    data_root = os.path.join(ROOT, 'data')
    file_path = os.path.join(data_root, 'pa3_{}.csv'.format(data_type))
    data = pd.read_csv(file_path, index_col=None)

    data_class = None
    if 'class' in data.columns:
        data_class = data['class'].copy()
        del data['class']

    return data, data_class


if __name__ == '__main__':
    import sys
    problems = [2]#[1, 2]
    if len(sys.argv) > 1:
        problems = [int(sys.argv[1])]

    data, data_class = load_data('train')
    val, val_class = load_data('val')
    if 1 in problems:

        depths = np.arange(1, 9)
        rez = pd.DataFrame(index=depths, columns=['Training', 'Validation'])

        for depth in depths:
            tree = BinaryDecisionTree(max_depth=depth)
            tree.train(data, data_class)

            rez.loc[depth, 'Training'] = tree.validate(data, data_class)[1] * 100
            rez.loc[depth, 'Validation'] = tree.validate(val, val_class)[1] * 100

        print(rez)

        rez.plot()
        plt.title('Decision Tree - Accuracy vs. Depth')
        plt.xlabel('Depth')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(os.path.join(ROOT, 'decision_tree.png'))

    if 2 in problems:

        ############
        # Problem 2
        ############
        print("2b")
        x = [1, 2, 5, 10, 25]
        y_train = []
        y_val = []
        for n in x:
            forest = RandomForest(n, 5, 2)
            forest.train(data, data_class)
            _, accuracy = forest.validate(data, data_class)
            y_train+=[accuracy]
            _, accuracy = forest.validate(val, val_class)
            y_val+=[accuracy]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y_train, label="train")
        ax1.plot(x, y_val, label="validate")
        ax1.legend()
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('n')
        plt.title('Random Forest: n')
        plt.show()

        print("2d")
        x = [1, 2, 5, 10, 25, 50]
        y_train = []
        y_val = []
        for m in x:
            forest = RandomForest(15, m, 2)
            forest.train(data, data_class)
            _, accuracy = forest.validate(data, data_class)
            y_train+=[accuracy]
            _, accuracy = forest.validate(val, val_class)
            y_val+=[accuracy]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y_train, label="train")
        ax1.plot(x, y_val, label="validate")
        ax1.legend()
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('m')
        plt.title('Random Forest: m')
        plt.show()

        print("2e")
        train_accuracy=0
        validation_accuracy=0
        for i in range(10):
            forest = RandomForest(1, 50, 2, useSeed=True)
            forest.train(data, data_class)
            _, accuracy = forest.validate(data, data_class)
            print('%s%f%s' %("Training accuracy: ", accuracy*100, "%"))
            train_accuracy+=accuracy
            _, accuracy = forest.validate(val, val_class)
            print('%s%f%s' %("Validation accuracy: ", accuracy*100, "%"))
            validation_accuracy+=accuracy
        print('%s%f%s' %("Average training accuracy: ", train_accuracy*10, "%"))
        print('%s%f%s' %("Average validation accuracy: ", validation_accuracy*10, "%"))
        


