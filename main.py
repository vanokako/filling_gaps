import preprocess as pp
import numpy as np
import neural_network as nn
import regression as reg
import pandas as pd
def create_table( data, indicies,*results):
    dataset = []
    for i in indicies:
        base = data[i[0]][i[1]]
        tmp = [base]
        for result in results:
            ratio = float('{:.3f}'.format(result[i[0]][i[1]]*100/base))
            tmp += [result[i[0]][i[1]], ratio]
        dataset.append(tmp)
    dataset = np.array(dataset)
    sums = np.sum(dataset, axis=0)
    dataset = np.vstack([dataset, sums])
    dataset = pd.DataFrame(dataset, columns=['base_value', 'first_network',
                                    'first_network, %','second_network',
                                    'second_network, %', 'ensemble',
                                    'ensemble, %','linear_regression',
                                    'linear_regression, %'])
    dataset.to_excel('./result.xlsx')

def normalization(data):
    columns = data.shape[1]
    coefs = []
    for i in range(columns):
        minimal = min(data[:,i])
        maximal = max(data[:,i])
        data[:,i] = (data[:,i]-minimal)/(maximal-minimal)
        coefs.append({'min': minimal, 'max':maximal})
    return data, coefs

def reverse_transformation(data, coefs):
    columns = data.shape[1]
    for i in range(columns):
        data[:,i] = data[:,i]*(coefs[i]['max'] - coefs[i]['min']) + coefs[i]['min']
    return data

def fill_gaps(data, predictions):
    for row in data:
        if np.any(np.isnan(row)):
            row[np.isnan(row)] = predictions.pop(0)
    data = reverse_transformation(data, coefs)
    return data

if __name__ == "__main__":
    data = pp.get_data('Glass.csv')
    data, coefs = normalization(data)
    predictions = []
    predictions_rbf = []
    predictions_ensemble = []
    data_with_gaps, data_without_gaps, gaps = pp.make_missing_value(data,
                                                            del_fraction_row = 0.01)
    missed_values = np.argwhere(np.isnan(data_with_gaps))
    copied_data = np.copy(data_with_gaps)
    copied_data_rbf = np.copy(data_with_gaps)
    copied_data_regression = np.copy(data_with_gaps)
    copied_data_ensemble = np.copy(data_with_gaps)
    for i in range(gaps.shape[0]):
        first_pred, second_pred = nn.make_prediction(data_without_gaps, gaps[i])
        predictions.append(first_pred)
        predictions_rbf.append(second_pred)
        predictions_ensemble.append((first_pred+second_pred)/2)
    predictions_lr = reg.predict(gaps, data_without_gaps)
    data = reverse_transformation(data, coefs)
    copied_data = fill_gaps(copied_data, predictions)
    copied_data_rbf = fill_gaps(copied_data_rbf, predictions_rbf)
    copied_data_regression = fill_gaps(copied_data_regression, predictions_lr)
    copied_data_ensemble = fill_gaps(copied_data_ensemble, predictions_ensemble)
    create_table(data, missed_values, copied_data, copied_data_rbf, copied_data_ensemble, copied_data_regression)
    print(np.isnan(copied_data).sum())
