# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Functions for estimating time-lag using AMI and extending it to mining gradual patterns.
The average mutual information I(X; Y) is a measure of the amount of “information” that
the random variables X and Y provide about one another.


"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

from .t_graank import TGrad


class TGradAMI(TGrad):

    def __init__(self, f_path: str, eq: bool, min_sup: float, target_col: int, err: float, num_cores: int):
        """"""
        # Compute MI w.r.t. target-column with original dataset to get the actual relationship
        # between variables. Compute MI for every time-delay/time-lag: if the values are
        # almost equal to actual, then we have the most accurate time-delay. Instead of
        # min-representativity value, we propose error-margin.

        super(TGradAMI, self).__init__(f_path, eq, min_sup, target_col=target_col, min_rep=0.25, num_cores=num_cores)
        self.error_margin = err
        self.min_membership = 0.001
        self.tri_mf_data = None  # The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        self.feature_cols = np.setdiff1d(self.attr_cols, self.target_col)
        self.initial_mutual_info = None
        self.mi_arr = None

    def compute_mutual_info(self):
        """"""
        # 1. Compute all the MI for every time-delay and store in list
        mi_list = []
        for step in range(self.max_step):
            attr_data, _ = self.transform_data(step)
            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[self.feature_cols], dtype=float).T
            mutual_info = mutual_info_regression(x_data, y)
            mi_list.append(mutual_info)
        mi_arr = np.array(mi_list, dtype=float)

        # 2. Standardize MI array
        # We replace 0 with -1 because 0 indicates NO MI, so we make it useless by making it -1, so it allows small
        # MI values to be considered and not 0. This is beautiful because if initial MI is 0, then both will be -1
        # making it the optimal MI with any other -1 in the time-delayed MIs
        mi_arr[mi_arr == 0] = -1
        # print(f"{mi_arr}\n")
        self.initial_mutual_info = mi_arr[0]  # step 0 is the MI without any time delay (or step)
        self.mi_arr = mi_arr[1:]

    def gather_delayed_data(self, optimal_dict: dict, max_step: int):
        """"""
        delayed_data = None
        time_data = []
        # time_dict = {}
        n = self.row_count
        k = (n - max_step)  # No. of rows created by largest step-delay
        for col_index in range(self.col_count):
            if (col_index == self.target_col) or (col_index in self.time_cols):
                # date-time column OR target column
                temp_row = self.full_attr_data[col_index][0: k]
            else:
                # other attributes
                step = optimal_dict[col_index]
                temp_row = self.full_attr_data[col_index][step: n]
                _, time_diffs = self.get_time_diffs(step)

                # Get first k items for delayed data
                temp_row = temp_row[0: k]

                # Get first k items for time-lag data
                temp_diffs = [(time_diffs[i]) for i in range(k)]
                time_data.append(temp_diffs)

                # for i in range(k):
                #    if i in time_dict:
                #        time_dict[i].append(time_diffs[i])
                #    else:
                #        time_dict[i] = [time_diffs[i]]
                # print(f"{time_diffs}\n")
                # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
            delayed_data = temp_row if (delayed_data is None) \
                else np.vstack((delayed_data, temp_row))

        # print(f"{time_dict}\n")
        return delayed_data, np.array(time_data)

    def discover_tgp(self, parallel=False):
        """"""

        # 1. Compute mutual information
        self.compute_mutual_info()

        # 2. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(self.mi_arr, self.initial_mutual_info))
        absolute_error = np.sqrt(squared_diff)
        optimal_steps_arr = np.argmin(absolute_error, axis=0)
        max_step = (np.max(optimal_steps_arr) + 1)
        print(f"Largest step delay: {max_step}\n")
        # print(f"Abs.E.: {absolute_error}\n")

        # 3. Integrate feature indices with the computed steps
        # optimal_dict = dict(map(lambda key, val: (int(key), int(val+1)), self.feature_cols, optimal_steps_arr))
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        print(f"Optimal Dict: {optimal_dict}\n")

        # 4. Create final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)
        # print(f"{delayed_data}\n")
        print(f"{time_data}\n")

        # 5. Build triangular MF
        a, b, c = TGradAMI.build_mf(time_data)
        self.tri_mf_data = np.array([a, b, c])
        print(f"{a}, {b}, {c}")

        # 6. Learn the best MF through slide-descent/sliding
        # 7. Apply cartesian product on multiple MFs to pick the MF with the biggest center (inference logic)

    @staticmethod
    def build_mf(time_data: np.ndarray):
        """"""
        # Reshape into 1-column dataset
        time_data = time_data.reshape(-1, 1)

        # Standardize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(time_data)

        # Apply SVD
        u, s, vt = np.linalg.svd(data_scaled, full_matrices=False)

        # Plot singular values to help determine the number of clusters
        # Based on the plot, choose the number of clusters (e.g., 3 clusters)
        num_clusters = int(s[0])

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data_scaled)

        # Get cluster centers
        centers = kmeans.cluster_centers_.flatten()

        # Define membership functions to ensure membership > 0.5
        # mf_list = []
        largest_mf = [0, 0, 0]
        for center in centers:
            half_width = 0.5 / 2  # since membership value should be > 0.5
            a = center - half_width
            b = center
            c = center + half_width
            if abs(c - a) > abs(largest_mf[2] - largest_mf[0]):
                largest_mf = [a, b, c]
            # mf_list.append((a, b, c))

        # Reverse the scaling
        a = scaler.inverse_transform([[largest_mf[0]]])[0, 0]
        b = scaler.inverse_transform([[largest_mf[1]]])[0, 0]
        c = scaler.inverse_transform([[largest_mf[2]]])[0, 0]
        return a, b, c

    @staticmethod
    def learn_best_mf(a: float, b: float, c: float, x_data: np.ndarray):
        """"""
        # if a <= x <= b then y_hat = (x - a) / (b - a)
        # if b <= x <= c then y_hat = (c - x) / (c - b)
        # Initialize parameters
        # min_membership = 0.001

        # 1. ML Approach
        tri_mf_data = np.array([a, b, c])
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
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_train = (x_data - x_min) / (x_max - x_min)
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
        bias = bias_hat * (x_max - x_min) + x_min
        print(f"bias: {bias}")

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
    def cost_function(y_true: np.ndarray, x_hat: np.ndarray, tri_mf: np.ndarray, min_membership: float):
        """
        Computes the logistic regression cost function for a fuzzy set created from a
        triangular membership function.

        :param y_true: A numpy array of the true labels.
        :param x_hat: A numpy array of the predicted labels.
        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
        :return: cost function values.
        """
        a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]
        # candidate_steps = [float((((0.5*(b+a)/x)-1) if (x <= b) else float((0.5*(b+c)/x)-1)) * x) for x in x_data]

        # Convert numpy array to tensor
        a_tensor = tf.constant(a, dtype=tf.float32)
        b_tensor = tf.constant(b, dtype=tf.float32)
        c_tensor = tf.constant(c, dtype=tf.float32)

        y_hat = tf.where(tf.logical_and(x_hat > a_tensor, x_hat < b_tensor), b_tensor-0.001, x_hat)
        y_hat = tf.where(tf.logical_and(y_hat > b_tensor, y_hat <= c_tensor), b_tensor+0.001, y_hat)
        y_hat = tf.where(y_hat <= a_tensor, a_tensor+0.001, y_hat)
        y_hat = tf.where(y_hat >= c_tensor, c_tensor-0.001, y_hat)
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


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
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
        return tf.matmul(inputs, self.fixed_weights) + self.bias
        # return tf.matmul(inputs, self.kernel) + self.bias
