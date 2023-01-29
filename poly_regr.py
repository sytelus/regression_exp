import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from scipy.special import softmax


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

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def poly_regress(ground_degree=4, model_degrees=4, data_len=20000, train_split=0.5, noise_level=0.1):
    train_count = int(data_len * train_split)
    test_count = data_len - train_count

    poly_coefficients = get_rand_poly_coefficients(ground_degree)

    x_all = get_rand_x(-5, 5, data_len)
    y_all = get_poly_y(x_all, poly_coefficients)

    x_train, x_test = x_all[:train_count], x_all[test_count:]
    x_train = add_noise(x_train, level=noise_level)

    y_train, y_test = y_all[:train_count], y_all[test_count:]
    y_train = add_noise(y_train, level=noise_level)

    model = np.polyfit(x_train, y_train, model_degrees)
    # Predict y values
    y_pred = np.polyval(model, x_test)

    # Calculate the loss (mean squared error)
    loss = mean_squared_error(normalize(y_test), normalize(y_pred), squared=False)
    coef_loss = 0 #np.mean(np.abs(poly_coefficients - model))

    return loss, x_test, y_test, y_pred, coef_loss, poly_coefficients, model


def distillation(ground_degree=10, teacher_degress=7, student_degrees=4, data_len=2000, train_split=0.5, noise_level=0.1):
    train_count = int(data_len * train_split)
    test_count = data_len - train_count

    poly_coefficients = get_rand_poly_coefficients(ground_degree)
    x_all = get_rand_x(-5, 5, data_len)
    y_all = get_poly_y(x_all, poly_coefficients)

    x_train, x_test = x_all[:train_count], x_all[test_count:]
    x_train = add_noise(x_train, level=noise_level)

    y_train, y_test = y_all[:train_count], y_all[test_count:]
    y_train = add_noise(y_train, level=noise_level)

    teacher_model = np.polyfit(x_train, y_train, teacher_degress)
    y_pred = np.polyval(teacher_model, x_test)
    teacher_loss = mean_squared_error(normalize(y_test), normalize(y_pred), squared=False)

    student_model = np.polyfit(x_train, y_train, student_degrees)
    y_pred = np.polyval(student_model, x_test)
    student_loss = mean_squared_error(normalize(y_test), normalize(y_pred), squared=False)

    student_distil_model = np.polyfit(x_train, np.polyval(teacher_model, x_train), student_degrees)
    y_pred = np.polyval(student_distil_model, x_test)
    student_distil_loss = mean_squared_error(normalize(y_test), normalize(y_pred), squared=False)

    return teacher_loss, student_loss, student_distil_loss