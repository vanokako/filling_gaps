import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np


def build_model(input_shape):
    model = Sequential()
    model.add(Dense(7, input_shape = (input_shape,),activation='relu', use_bias=False))
    model.add(Dense(1, activation='relu', use_bias=False))
#    model.compile(loss='mae', optimizer='adam')
    #ts_inputs = tf.keras.Input(shape=(input_shape, ))
#    x = LSTM(units=10, return_sequences = False, )
#    x = Dropout(0.2)(x)
#    outputs = Dense(1, activation='linear')(x)
#    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
#    model = Sequential()
#    model.add(LSTM(units=10, return_sequences = False, input_shape=(input_shape, 1)))
#    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss=['mae'],
                  metrics=['mae'])
    return model

def make_prediction(data, gaps, min, max):
    batch_size = 15
    x, y, predict_data = prepare_for_traning(data, gaps)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        mode='min')
    print(predict_data)
#    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
#    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
    input_shape = data.shape[1]
    output_length = data.shape[0]
    model = build_model(input_shape)
    history = model.fit(x_train, y_train,
                        epochs=1000,
                        validation_data=(x_test, y_test),
                        verbose = 0,
                        batch_size=10
#                        callbacks=[early_stopping]
                        )
    a = model.predict(x_train)
    print(model.get_weights())
    print(y_train)
    print(a)
    print(y_train*(max-min)+min)
    print(a*(max-min)+min)

def prepare_for_traning(data, gaps):
    x = []
    predict_data = []
    y = []
    for i,elem in enumerate(gaps):
        if np.isnan(elem):
            predict_data.append(data[i])
        else:
            y.append(elem)
            x.append(data[i])
    return np.array(x), np.array(y), np.array(predict_data)
