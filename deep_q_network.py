import tensorflow as tf
from keras.layers import Input, Dense

class DeepQNetwork():
    def __init__(self, input_shape, action_space, batch_size):
        self.batch_size = batch_size
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=512, input_shape=input_shape, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=action_space, activation='softmax'))
        self.model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

    def train(self, states, targets):
        self.model.fit(states, targets, batch_size=self.batch_size, verbose=0)

    def predict(self, states):
        prediction_prob = self.model.predict(states)
        return prediction_prob

    def copy_weights_from_nn(self, neural_network):
        self.model.set_weights(neural_network.model.get_weights())

    def save(self):
        self.model.save("dqn.h5")
