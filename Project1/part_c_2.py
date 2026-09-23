from sklearn.utils import resample
from part_a import *

def bootstrap_ols(x, y, k, degree, ts = 0.2, rs = 2026):
    scaler = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ts, random_state = rs)
    X_test = design_matrix(x_test, degree)
    predictions = np.zeros((k, len(x_test) ))

    for i in range(k):
        x_b, y_b = resample(x_train, y_train)
        X_b = design_matrix(x_b, degree)
        X_b_scaled = scaler.fit_transform(X_b)
        X_test_scaled = scaler.transform(X_test)
        y_b_centered = y_b - y_b.mean()

        theta = OLS(X_b_scaled, y_b_centered)
        y_tilde = X_test_scaled @ theta
        predictions[i, :] = y_tilde + y_b.mean()
    

    y_tilde_mean = np.mean(predictions, axis = 0, keepdims = True)
    error = np.mean(np.mean((y_test - predictions) ** 2, axis = 0, keepdims = True))
    variance = np.mean(np.var(predictions, axis = 0, keepdims = True))
    bias = np.mean((y_tilde_mean - y_test) ** 2)


    return error, bias, variance

def plot_bias_var(x, y, k, degree_max, n):

        powers = np.arange(1, degree_max + 1)
        mses = np.zeros(degree_max)
        biases = np.zeros(degree_max)
        variances = np.zeros(degree_max)

        for p in range(1, degree_max + 1):
             m, b, v = bootstrap_ols(x, y, k, p)

             mses[p - 1] = m
             biases[p - 1] = b
             variances[p - 1] = v
            
        plt.plot(powers, mses, "o-", label = "$MSE$")
        plt.plot(powers, biases, "o-", label = "$Bias^2 + \\sigma^2$")
        plt.plot(powers, variances, "o-", label = "$Var$")
        plt.title(f"Mean statistics over {k} bootstraps, for {n} datapoints")
        plt.yscale("log")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    n = 100
    x, y = runge_data(n = n)
    k = 100
    degree_max = 15

    plot_bias_var(x, y, k, degree_max, n)