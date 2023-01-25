import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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


data_len, train_split = 2000, 1000
test_split = data_len - train_split

poly_coefficients = get_rand_poly_coefficients(10)

x_all = get_rand_x(-5, 5, data_len)
x_train, x_test = x_all[:train_split], x_all[test_split:]
x_train = add_noise(x_train, level=0.1)

y_all = get_poly_y(x_all, poly_coefficients)
y_train, y_test = y_all[:train_split], y_all[test_split:]
y_train = add_noise(y_train, level=0.1)

model = np.polyfit(x_train, y_train, 10)
# Predict y values
y_pred = np.polyval(model, x_test)

# Calculate the loss (mean squared error)
loss = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", loss)
print("Y stats:", np.min(y_test), np.max(y_test), np.mean(y_test), np.std(y_test))

#print(x_test[:10], y_test[:10], y_pred[:10])