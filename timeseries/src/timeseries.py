from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


data = list(csv.reader(open("../data/daily-min-temps.csv")))[1:]
time = [d[0] for d in data]
series = [float(d[1]) for d in data]

# plot_series(time, series)

split_time = int(len(data) * 0.8)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# plot_series(time, series)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset.element_spec)
for x, y in dataset.take(1):
    print(x.numpy().shape)
    print(y.numpy().shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9),
    metrics=['mse']
)

# find out good learning rate
# lrsc = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
# history = model.fit(dataset, epochs=100, callbacks=[lrsc], verbose=2)
# learning_rates = 1e-8 * 10 ** (np.arange(100) / 20)
# plt.semilogx(learning_rates, history.history['loss'])
# plt.show()

# let's train for real
history = model.fit(dataset, epochs=100, verbose=2)

ds = tf.data.Dataset.from_tensor_slices(x_valid)
ds = ds.window(window_size, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda window: window.batch(window_size))
ds = ds.batch(batch_size).prefetch(1)

results = model.predict(ds)
results = np.array(results)
results = results[:, 0]

error = keras.metrics.mean_absolute_error(x_valid[:-window_size+1], results).numpy()
print(error)

plt.figure(figsize=(10, 6))

plot_series(time_valid[:-window_size+1], x_valid[:-window_size+1])
plot_series(time_valid[:-window_size+1], results)

plt.show()

