from sklearn.linear_model import LinearRegression
import numpy as np
def predict(gaps, data):
    predictions =[]
    for row in gaps:
        idxs_of_nan = np.argwhere(np.isnan(row)).ravel()
        idxs_not_nan = np.argwhere(~np.isnan(row)).ravel()
        print(data[:,idxs_not_nan])
        reg = LinearRegression().fit(data[:,idxs_not_nan], data[:,idxs_of_nan])
        predictions.append(reg.predict(row[idxs_not_nan].reshape((1,-1))).tolist()[0])
    return predictions
