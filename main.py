from poly_regr import poly_regress
import numpy as np

def main():
    loss, x_test, y_test, y_pred, coef_loss, poly_coefficients, model = poly_regress(ground_degree=2)

    print("Loss:", loss, np.min(y_test), np.max(y_test), np.mean(y_test), np.std(y_test))
    print('Coefficient loss:', coef_loss, np.min(poly_coefficients), np.max(poly_coefficients), np.mean(poly_coefficients), np.std(poly_coefficients))

if __name__ == "__main__":
    main()