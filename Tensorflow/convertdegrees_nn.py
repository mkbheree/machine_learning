import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10,  0,  8, 15, 22,  38])
fahren = np.array([-40,  14, 32, 46, 59, 72, 100])

layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential(layer0)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius, fahren, epochs=500, verbose=False)


print(model.predict([100.0]))
print("These are the layer variables: {}".format(layer0.get_weights()))

'''
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
'''
