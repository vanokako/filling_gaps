import preprocess as pp
import numpy as np
import neural_network as nn
from sklearn.preprocessing import MinMaxScaler
if __name__ == "__main__":
    data = pp.get_data('Boston.csv')
    # st_el = [float(i) for i in range(1, 50)]
    # nd_el = [float(i+1) for i in range(1, 50)]
    # pred_el = [float(st_el[i]+nd_el[i]*5) for i in range(49)]
    # data = np.array([pred_el,st_el, nd_el]).T
    min = data.min()
    max = data.max()
    data = (data-min)/(max-min)

    print(data.shape)
    data_with_gaps, data_without_gaps, gaps = pp.make_missing_value(data, del_fraction=0.15)
    for i in range(gaps.shape[1]):
        nn.make_prediction(data_without_gaps, gaps[i], max, min)
