import numpy as np
import pandas as pd
import os
import sys

def load_data(data_type, preprocess=True, normalize=True, statistics=False, against=None):
    """
    Loads data in from the data files.
    :param data_type: A string corresponding to a file in the data folder. So 'dev', 'test', or 'train'
    :param preprocess: If True, takes out the ID and splits up the date into year/month/date
    :param normalize: Scales the columns to be between 0 and 1.
    :param against: Another string like in data_type specifying a dataset to normalize against.
                    If None, normalizes against itself.
    :return: A DataFrame
    """

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    file_path = os.path.join(data_root, 'PA1_{}.csv'.format(data_type))
    df = pd.read_csv(file_path, index_col=None)

    if preprocess:
        del df['id']
        dates = pd.to_datetime(df['date'])
        df['year'] = dates.map(lambda x: x.year)
        df['month'] = dates.map(lambda x: x.month)
        df['day'] = dates.map(lambda x: x.day)
        del df['date']

    if statistics:
        print("statistics table")
        # Numeric features
        print("feature \t mean \t standard deviation \t range")
        for key in ["year", "month", "day","bedrooms","bathrooms","sqft_living","sqft_lot","floors","view","sqft_above",
            "sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]:
            print( key + "\t %f \t %f \t %f" %(np.mean(df[key]), np.std(df[key]), np.ptp(df[key])))
        # categorical features
        for key in ["waterfront","condition","grade"]:
            unique, counts = np.unique(df[key], return_counts=True)
            d = dict(zip(unique, counts))
            printstr = key + "\t"
            num_samples = len(df[key])
            for category in d.keys():
                printstr += str(category) + ": " + str(float(d[category])/num_samples) + "\t"
            print(printstr)

    if normalize:

        normalize_df = df
        if against is not None:
            normalize_df = load_data(against, preprocess=True, normalize=False)

        cols = [col for col in df.columns if col != 'price']

        mins = normalize_df[cols].min()
        maxs = normalize_df[cols].max()
        df[cols] = (df[cols] - mins) / (maxs - mins)

    df['dummy'] = 1.0

    return df

def convert_df_to_latex_table(df):

    header = 'Stat & {} \\\\ \hline'.format(' & '.join(df.columns))
    body = ' \\\\\n'.join(
        [str(idx).replace('_', '\\_') + ' & ' + ' & '.join(['{:.2f}'.format(stat) for stat in df.loc[idx]]) for idx in
         df.index])

    return header + body



if __name__ == '__main__':

    try:
        to_run = sys.argv[1]
    except IndexError:
        to_run = None

    if to_run == '0':
        # Outputting table for Latex
        training_data = load_data('train', normalize=False)
        summ = training_data.describe().T
        del summ['count']
        summ['range'] = summ['max'] - summ['min']
        summ = summ[['mean', 'std', 'range']]
        with open('latextable.txt', 'w') as fh:
            fh.write(convert_df_to_latex_table(summ))

        for idx in summ.index:
            unique_vals = training_data[idx].unique()
            if len(unique_vals) < 20:
                print('{} looks categorical'.format(idx))
                gp = training_data[idx].groupby(training_data[idx]).count()

                for val in sorted(unique_vals):
                    print('\t{}: {:.1f}%'.format(val, 100 * gp[val] / float(gp.sum())))

        sys.exit(0)


    training_data = load_data('train')
    validation_data = load_data('dev', against='train')
    regressors = [x for x in training_data.columns if x != 'price']

    from lin_reg import LinearRegressor
    import matplotlib.pyplot as plt

    ############
    # Problem 1
    ############
    if to_run == '1':
        learning_rates = [10**(-1*x) for x in range(8)]
        for learning_rate in learning_rates:

            # useless_regressors = ['day', 'month', 'year', 'zipcode']


            # Set up the linear regression model with our desired learning rate/regularization params
            reg = LinearRegressor(regressors, 'price', learning_rate, regularize=0)

            # Train it with a given convergence threshold and number of iterations (can set to None if you want)
            reg.train(training_data, 0.5, max_iter=40000)

            # Testing if it worked
            print("Weights:")
            print(reg.w)
            print()

            y = reg.sseData
            iterations = reg.sseData.size
            print(str(iterations) + " iterations for convergence")
            print("Training loss: " + str(reg.compute_loss(training_data)))
            x = range(0, iterations)

            validation_data = load_data('dev', against='train')
            print("Validation loss: " + str(reg.compute_loss(validation_data)))

            print()
            print()

            plt.scatter(x, y)
            plt.xlabel("Iterations")
            plt.ylabel("Average SSE")
            plt.title("SSE vs. Iterations for Learning Rate " + str(learning_rate))
            plt.show()

        sys.exit(0)


    ############
    # Problem 2
    ############

    if to_run == '2':
        lambdas = [0, 10**(-3), 10**(-2), 0.1, 1, 10, 100]
        sses = pd.DataFrame(index=lambdas, columns=['Training', 'Validation'])
        for l in lambdas:

            useless_regressors = []
            regressors = [x for x in training_data.columns if x != 'price' and x not in useless_regressors]

            # Set up the linear regression model with our desired learning rate/regularization params
            reg = LinearRegressor(regressors, 'price', 0.01, regularize=l)

            # Train it with a given convergence threshold and number of iterations (can set to None if you want)
            reg.train(training_data, 0.5, max_iter=10000)

            # Testing if it worked
            print("Weights:")
            print(reg.w)
            print()

            y = reg.sseData
            iterations = reg.sseData.size

            print(str(iterations) + " iterations for convergence")

            validation_data = load_data('dev', against='train')

            training_loss = reg.compute_loss(training_data, for_reporting=True)
            validation_loss = reg.compute_loss(validation_data, for_reporting=True)

            sses.loc[l, 'Training'] = training_loss
            sses.loc[l, 'Validation'] = validation_loss

            print("Training SSE: " + str(training_loss))
            print("Validation SSE: " + str(validation_loss))

            print()
            print()

        with open('part2sses.txt', 'w') as fh:
            fh.write(convert_df_to_latex_table(sses))

        sys.exit(0)


    ############
    # Problem 3
    ############

    if to_run == '3':
        training_data = load_data('train', normalize=False)
        validation_data = load_data('dev', normalize=False)

        labels = ['1', '1e-3', '1e-6', '1e-9', '1e-15', '0']
        learning_df = pd.DataFrame(index=np.arange(10000), columns=labels)
        validation_df = learning_df.copy()

        seed_weights = None

        for label in labels:
            learning_rate = float(label)
            reg = LinearRegressor(regressors, 'price', learning_rate, regularize=0, weight_range=0.000001,
                                  validation_set=validation_data)

            if seed_weights is None:
                seed_weights = reg.w.copy()
            else:
                reg.w = seed_weights.copy()

            reg.train(training_data, 0.5, max_iter=10000)

            learning_df[label] = reg.reportingSse
            validation_df[label] = reg.validationSse

        sys.exit(0)

    ############
    # Prediction
    ############

    if to_run == 'predict':

        to_predict = load_data('test', against='train')
        reg = LinearRegressor(regressors, 'price', 0.001, regularize=0.001, weight_range=0.00001)

        # Train it with a given convergence threshold and number of iterations (can set to None if you want)
        reg.train(training_data, 0.5, max_iter=40000)
        pred_vals = reg.predict(to_predict)

        with open('prediction_values.txt', 'w') as fh:
            fh.write('\n'.join([str(v) for v in pred_vals]))

        print('Predicted values output!')
        sys.exit(0)

    raise ValueError('Please run the script with 0, 1, 2, 3, or "predict" as the argument.')