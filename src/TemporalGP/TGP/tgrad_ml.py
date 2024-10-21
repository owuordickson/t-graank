# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
A module that applies a Neural Network to learn the best location (or slide) of a triangular membership function.
The best slide is modelled as the bias, so we keep the weights constant and try to learn the bias.
"""


import numpy as np
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.fixed_weights = None
        self.bias = None
        self.kernel = None
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer='ones',
                                      trainable=False)
        # Initialize bias as a trainable variable
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        # Set weights to 1 and make them non-trainable
        self.fixed_weights = tf.ones((input_shape[-1], self.units), dtype=tf.float32)

    def call(self, inputs):
        # print(f"y_hat bias: {self.bias}")
        return tf.matmul(inputs, self.fixed_weights) + self.bias
        # return tf.matmul(inputs, self.kernel) + self.bias

    @staticmethod
    def learn_best_mf_ann(a: float, b: float, c: float, x_data: np.ndarray):
        """"""
        # if a <= x <= b then y_hat = (x - a) / (b - a)
        # if b <= x <= c then y_hat = (c - x) / (c - b)
        # Initialize parameters
        # min_membership = 0.001

        # 1. ML Approach
        # tri_mf_data = np.array([a, b, c])
        # Normalization
        # combined_data = np.concatenate((tri_mf_data, x_data))
        # x_min = np.min(combined_data)
        # x_max = np.max(combined_data)
        # Normalize x_train and tri_mf_data
        # tri_mf_data = (tri_mf_data - x_min) / (x_max - x_min)
        # x_train = (x_data - x_min) / (x_max - x_min)
        # y_train = np.full_like(x_data, 1, dtype=float)

        y_train = np.where(np.logical_and(x_data > a, x_data < b), b - 0.001, x_data)
        y_train = np.where(np.logical_and(y_train > b, y_train <= c), b + 0.001, y_train)
        y_train = np.where(y_train <= a, a + 0.001, y_train)
        y_train = np.where(y_train >= c, c - 0.001, y_train)
        # y_train = x_data + 1.5

        # Normalize x_train
        x_data = x_data.reshape(-1, 1)
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_data)
        # x_train = np.array(x_data, dtype=float)

        print(f"x-train: {x_train}")
        print(f"y-train: {y_train}")
        # print(f"tri-mf: {tri_mf_data}")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            BiasLayer(1)
        ])

        # Automated
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer=opt, loss='mse')
        print(model.summary())
        model.fit(x_train, y_train, epochs=50)

        # Custom Training Loop
        # optimizer = tf.keras.optimizers.Adam()
        # epochs = 10
        # for epoch in range(epochs):
        #    with tf.GradientTape() as tape:
        #        predictions = model(x_train, training=True)
        #        loss = TGradAMI.cost_function(y_train, predictions, tri_mf_data, min_membership)
        #    gradients = tape.gradient(loss, model.trainable_variables)
        #    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

        bias_hat = model.layers[0].bias.numpy()
        bias = scaler.inverse_transform([[float(bias_hat)]])[0, 0]
        print(f"bias: {bias}")
        # print(model.layers[0].get_weights())
        # print(f"weights: {model.layers[0].kernel.numpy()}")
        # print(f"weights: {model.layers[0].fixed_weights.numpy()}")

        # 2. Manual Approach
        # candidate_steps = [float((((0.5*(b+a)/x)-1) if (x <= b) else float((0.5*(b+c)/x)-1)) * x) for x in x_data]
        # print(f"Candidate Steps: {candidate_steps}")
        # for i in range(10):
        #    print(f"Slide: {i}")
        #    print(f"x-train: {x_train}")
        #    # 1. Generate fuzzy data set using MF from x_data
        #    # Method 1 (OK)
        #    x_hat = np.where(x_train <= b, (x_train-a)/(b-a), (c-x_train)/(c-b))
        #    b = 0.5
        #    x_train = x_train + b
        #    # 2. Generate y_train based on the given criteria (x>0.5)
        #    y_hat = np.where((x_hat >= min_membership), 1, 0)
        #    print(f"x-hat: {x_hat}")
        #    print(f"y-hat: {y_hat}\n")

    @staticmethod
    def cost_function_wrapper(tri_mf: np.ndarray, min_membership: float):
        """
        Computes the logistic regression cost function for a fuzzy set created from a
        triangular membership function.

        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
        :return: cost function values.
        """
        a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]

        def custom_loss(y_true: np.ndarray, x_hat: np.ndarray):
            """
            Computes the logistic regression cost function for a fuzzy set created from a
            triangular membership function.

            :param y_true: A numpy array of the true labels.
            :param x_hat: A numpy array of the predicted labels.
            :return: loss values.
            """

            # Convert numpy array to tensor
            a_tensor = tf.constant(a, dtype=tf.float32)
            b_tensor = tf.constant(b, dtype=tf.float32)
            c_tensor = tf.constant(c, dtype=tf.float32)

            # 1. Generate fuzzy data set using MF from x_data
            y_hat = tf.where(x_hat <= b_tensor,
                             (x_hat - a_tensor) / (b_tensor - a_tensor),
                             (c_tensor - x_hat) / (c_tensor - b_tensor))

            # 2. Generate y_train based on the given criteria (x>minimum_membership)
            y_hat = tf.where(y_hat >= min_membership, 0.99, 0.01)

            # 3. Compute loss
            loss = tf.keras.losses.binary_crossentropy(y_true, y_hat)
            return loss

        return custom_loss

    @staticmethod
    def cost_function(y_true: np.ndarray, x_hat: np.ndarray, tri_mf: np.ndarray):
        """
        Computes the logistic regression cost function for a fuzzy set created from a
        triangular membership function.

        :param y_true: A numpy array of the true labels.
        :param x_hat: A numpy array of the predicted labels.
        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :return: cost function values.
        """
        a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]
        # candidate_steps = [float((((0.5*(b+a)/x)-1) if (x <= b) else float((0.5*(b+c)/x)-1)) * x) for x in x_data]

        # Convert numpy array to tensor
        a_tensor = tf.constant(a, dtype=tf.float32)
        b_tensor = tf.constant(b, dtype=tf.float32)
        c_tensor = tf.constant(c, dtype=tf.float32)

        y_hat = tf.where(tf.logical_and(x_hat > a_tensor, x_hat < b_tensor), b_tensor - 0.001, x_hat)
        y_hat = tf.where(tf.logical_and(y_hat > b_tensor, y_hat <= c_tensor), b_tensor + 0.001, y_hat)
        y_hat = tf.where(y_hat <= a_tensor, a_tensor + 0.001, y_hat)
        y_hat = tf.where(y_hat >= c_tensor, c_tensor - 0.001, y_hat)
        y_hat = tf.squeeze(y_hat)  # Ensure y_hat has the same shape as y_true
        print(f"x-hat Model: {y_hat}")

        # 1. Generate fuzzy data set using MF from x_data
        # y_hat = tf.where(x_hat <= b_tensor,
        #                 (x_hat - a_tensor) / (b_tensor - a_tensor),
        #                 (c_tensor - x_hat) / (c_tensor - b_tensor))
        # 2. Generate y_train based on the given criteria (x>minimum_membership)
        # y_hat = tf.where(y_hat >= min_membership, 0.9, 0.2)
        # y_hat = tf.squeeze(x_hat)  # Ensure y_hat has the same shape as y_true

        # 3. Compute loss
        loss = tf.keras.losses.binary_crossentropy(y_true, y_hat)
        return tf.reduce_mean(loss)
