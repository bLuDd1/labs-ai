import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_y(x):
    return x * np.cos(2 * x) + np.sin(x / 2)


def get_z(x, y):
    return np.sin(y) + np.cos(x / 2)


x_train = np.linspace(0, 16, 80)
x_test = np.linspace(16, 20, 20)
y_train = get_y(x_train)
y_test = get_y(x_test)
z_train = get_z(x_train, y_train)
z_test = get_z(x_test, y_test)


data_train = np.vstack((x_train, y_train)).T
data_test = np.vstack((x_test, y_test)).T
data_train_rnn = data_train.reshape(x_train.shape[0], 1, 2)
data_test_rnn = data_test.reshape(x_test.shape[0], 1, 2)


def train(model, data_train_val, z_train_val, data_test_val, z_test_val, epochs=100):
    history = model.fit(data_train_val, z_train_val, epochs=epochs, validation_data=(data_test_val, z_test_val), verbose=0)
    return history.history


def create_plot(history, name):
    print(f"Model name: {name}")
    print(f"Final training loss: {history['loss'][-1]:.10f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.10f}\n\n")
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label=f'{name} (training loss)')
    plt.plot(history['val_loss'], label=f'{name} (validation loss)')
    plt.title('Training & validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def create_feed_forward(neurons):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = train(create_feed_forward(10), data_train, z_train, data_test, z_test)
create_plot(model, 'Feed forward - 10 neurons')

model = train(create_feed_forward(20), data_train, z_train, data_test, z_test)
create_plot(model, 'Feed forward - 20 neurons')


def create_cascade_forward(neurons, layers=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    if layers == 2:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = train(create_cascade_forward(20), data_train, z_train, data_test, z_test)
create_plot(model, 'Cascade forward - 20 neurons')

model = train(create_cascade_forward(10, 2), data_train, z_train, data_test, z_test)
create_plot(model, 'Cascade forward - 10 neurons, 2 hidden layers')

def create_elman(neurons, layers=1):
    layers_array = [tf.keras.layers.Dense(1)]
    for _ in range(layers):
        layers_array.append(tf.keras.layers.SimpleRNN(neurons, activation='relu', return_sequences=True))
    layers_array.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers_array)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = train(create_elman(15), data_train_rnn, z_train, data_test_rnn, z_test)
create_plot(model, 'Elman - 15 neurons')

model = train(create_elman(5, 3), data_train_rnn, z_train, data_test_rnn, z_test)
create_plot(model, 'Elman - 5 neurons, 3 hidden layers')