import numpy as np
import pandas as pd
def get_data(filename):
    data = pd.read_csv(filename, delimiter=',')
    data = np.array(data.drop(data.columns[0],axis=1))
    print(data)
    return data


def make_missing_value(X, del_fraction=0.05, del_fraction_column=1.0, del_fraction_row=1.0, del_columns=None):
    N = X.shape[0]
    D = X.shape[1]

    col_count = int(D * del_fraction_column)
    row_count = int(N * del_fraction_row)

    del_columns = np.random.permutation(np.arange(D))[:col_count]

    if del_columns is None:
        del_columns = np.arange(D)[D-col_count:]
    del_row = np.random.permutation(np.arange(N))[:row_count]
    new_del_fraction = del_fraction / (del_fraction_row * del_fraction_column)

    if new_del_fraction > 1.0:
        new_del_fraction = 0.5
        print('Warning: del_fraction is too big for del_fraction_column and del_fraction_row. ' +
              'It will be set to {0}.'.format(0.5 * del_fraction_column * del_fraction_row))

    delete_mask = np.array(np.random.random((N, D)) < new_del_fraction, dtype=int)
    delete_mask[del_row, :] += 1
    delete_mask[:, del_columns] += 1
    delete_mask = np.array(delete_mask == 3, dtype=bool)

    new_X = np.array(X)
    # new_X[25][0] = np.nan
    new_X[delete_mask] = np.nan
    X_without_gaps = new_X[~np.isnan(new_X).any(axis=1)]
    gaps = new_X[np.isnan(new_X).any(axis=1)]
    return new_X, X_without_gaps, gaps
