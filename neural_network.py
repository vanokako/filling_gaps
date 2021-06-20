import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np



def build_model(input_shape, output_length):
    model=Sequential([

    tf.keras.layers.Dense(input_shape),
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
])
    model.compile(optimizer=Adam(learning_rate=0.005),
                  loss=['mae'],
                  metrics=['mse'])

    return model

def build_model_rbf(input_shape, output_length, X):
    model = Sequential()
    rbflayer = RBFLayer(input_shape,
                        initializer=InitCentersRandom(X),
                        betas=3.0,
                        input_shape=(1,))
    model.add(rbflayer)
    model.add(Dense(256))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(Dense(output_length))
    model.compile(loss='mae',
                  metrics=['mse'],
                  optimizer=Adam())
    return model

def make_prediction(data, gaps):

    x, y, predict_data = prepare_for_traning(data, gaps)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    batch_size = 4 if x_train.shape[0] // 16 <= 4 else x_train.shape[0] // 16
    predict_data = np.reshape(predict_data, (1, predict_data.shape[0]))
    print(x_test.shape, y_test.shape)
    print(x_train.shape, y_train.shape)


    best_model=tf.keras.callbacks.ModelCheckpoint(
                                    filepath='./checkpoint',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
    best_model_1 =tf.keras.callbacks.ModelCheckpoint(
                                    filepath='./checkpoint1',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
    input_shape = x_train.shape[1]
    output_length = np.isnan(gaps).sum()
    model = build_model(input_shape, output_length)
    model_rbf = build_model_rbf(input_shape, output_length, x)
    history_rbf = model_rbf.fit(x_train, y_train,
                        epochs=300,
                        validation_data=(x_test, y_test),
                        verbose = 0,
                        batch_size=batch_size,
                        callbacks=[best_model]
                        )

    model_rbf.load_weights('./checkpoint')
    history = model.fit(x_train, y_train,
                        epochs=300,
                        validation_data=(x_test, y_test),
                        verbose = 0,
                        batch_size=batch_size,
                        callbacks=[best_model_1]
                        )
    model.load_weights('./checkpoint1')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    results_rbf = model_rbf.evaluate(x_test, y_test, batch_size=batch_size)
    prediction = model.predict(predict_data)
    prediction_rbf = model_rbf.predict(predict_data)

    return [prediction.ravel(), prediction_rbf.ravel(), results, results_rbf]


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
