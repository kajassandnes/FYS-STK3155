from part_a import *

def plot_test_train_mse(data, target, ts = 0.2, degree_max = 15, method = OLS, rs = 2026):

    """
    Input data, target variable, training size, and maximum degree of polnomial.
    Plots mse, r2, and coefficients.
    """
    powers = np.arange(1, degree_max + 1)
    mses_train = np.zeros_like(powers, dtype = float)
    mses_test = np.zeros_like(powers, dtype = float)

    for p in range(1, degree_max + 1):
        X = design_matrix(data, p)
        X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

        theta = method(X_train_scaled, y_train_centered)
        y_tilde_train, y_tilde_test = X_train_scaled @ theta, X_test_scaled @ theta
        mse_score_train, mse_score_test = mse(y_train_centered, y_tilde_train), mse(y_test_centered, y_tilde_test)

        mses_train[p - 1] = mse_score_train
        mses_test[p - 1] = mse_score_test


    plt.plot(powers, mses_train, "o-", label = "Training MSE")
    plt.plot(powers, mses_test, "o-", label = "Test MSE")
    plt.xlabel("Powers")
    plt.ylabel("MSE")
    plt.title("MSE of training and test data as function of complexity")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    x, y = runge_data()
    plot_test_train_mse(x, y, degree_max = 17)