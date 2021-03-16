import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np


def build_model(input_shape, output_length):
    model=Sequential([

    tf.keras.layers.Dense(1024, input_shape = (input_shape,)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(units=256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(units=256),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(units=128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(units=output_length),
],name="Larger_network",)
    # model = Sequential()
    # model.add(Dense(16, input_shape = (input_shape,),activation='relu', use_bias=False))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation='relu'))
#    model.compile(loss='mae', optimizer='adam')
    #ts_inputs = tf.keras.Input(shape=(input_shape, ))
#    x = LSTM(units=10, return_sequences = False, )
#    x = Dropout(0.2)(x)
#    outputs = Dense(1, activation='linear')(x)
#    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
#    model = Sequential()
#    model.add(LSTM(units=10, return_sequences = False, input_shape=(input_shape, 1)))
#    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=['mae'],
                  metrics=['mse'])
    return model

def make_prediction(data, gaps, max, min):
    batch_size = 4
    x, y, predict_data = prepare_for_traning(data, gaps)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

    print(x_test.shape, y_test.shape)
    print(x_train.shape, y_train.shape)
    checkpoint_filepath = './checkpoint'
    best_model = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


    input_shape = x_train.shape[1]
    output_length = data.shape[1]-len(predict_data)
    model = build_model(input_shape, output_length)
    history = model.fit(x_train, y_train,
                        epochs=1000,
                        validation_data=(x_test, y_test),
                        verbose = 1,
                        batch_size=batch_size,
                        callbacks=[best_model]
                        )
    model.load_weights(checkpoint_filepath)
    print(history.history['val_loss'])
    a = model.predict(x_test)
    print("++++++++++++++TEST++++++++++")
    print(y_test[:5]*(max-min)+min)
    print("++++++++++++++PRREDICT++++++++++")
    print(a[:5]*(max-min)+min)

    # print(y_test)
    # print(a)

def prepare_for_traning(data, gaps):
    x = []
    indexes =[]
    predict_data = []
    y = []
    for i,elem in enumerate(gaps):
        if np.isnan(elem):
            indexes.append(i)
    predict_data = [gaps[j] for j in range(len(gaps)) if j not in indexes]
    for row in data:
        x.append([row[k] for k in range(len(row)) if k not in indexes])
        y.append([row[k] for k in range(len(row)) if k in indexes])
    return np.array(x), np.array(y), np.array(predict_data)
