import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass


def get_rand_x(min, max, num):
    # Generate x values
    return np.random.uniform(min, max, num)

def get_rand_poly_coefficients(degree):
    # Generate random coefficients for 7th degree polynomial
    return np.random.rand(degree + 1)

def get_poly_y(x, coefficients):
    return np.polyval(coefficients, x)

def add_noise(y, level):
    return y + (np.random.rand(y.shape[0]) - 0.5) * level * 2 * y

def poly_regress(ground_degree=10, data_len=2000, train_split=1000, noise_level=0.1):
    test_split = data_len - train_split

    poly_coefficients = get_rand_poly_coefficients(ground_degree)

    x_all = get_rand_x(-5, 5, data_len)
    x_train, x_test = x_all[:train_split], x_all[test_split:]
    x_train = add_noise(x_train, level=noise_level)

    y_all = get_poly_y(x_all, poly_coefficients)
    y_train, y_test = y_all[:train_split], y_all[test_split:]
    y_train = add_noise(y_train, level=noise_level)

    model = np.polyfit(x_train, y_train, ground_degree)
    # Predict y values
    y_pred = np.polyval(model, x_test)

    # Calculate the loss (mean squared error)
    loss = mean_squared_error(y_test, y_pred, squared=False)

    return loss, x_test, y_test, y_pred