from part_a import *


def ridge(X, target, lmbda = 0.1):
    """
    Input data, lmbda, and target vriable. return polynomial coefficients.
    Uses ridge regression with np.linalg.solve().
    """
    n, p = X.shape
    I = np.eye(p)

    theta = np.linalg.solve(X.T @ X + lmbda * I, X.T @ target)
    return theta


def plot_score_lmbdas(data, target, pow,  ts, rs, lmbda_min = -8, lmbda_max = 2, nlmbdas = 20):
    """
    Input data, target variable, power, training size, and lmbda values.
    Plots mse, r2, and coefficients.
    """
    lmbdas = np.logspace(lmbda_min, lmbda_max, nlmbdas)
    mses = np.zeros_like(lmbdas, dtype = float)
    R2s = np.zeros_like(lmbdas, dtype = float)
    theta_norms = np.zeros_like(lmbdas, dtype = float)

    for p in range(nlmbdas):
        X = design_matrix(data, pow)
        X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

        theta = ridge(X_train_scaled, y_train_centered, lmbda = lmbdas[p])
        y_tilde = X_test_scaled @ theta
        mse_score, R2_score = mse(y_test_centered, y_tilde), r2_score(y_test_centered, y_tilde)

        mses[p] = mse_score
        R2s[p] = R2_score
        theta_norms[p] = np.linalg.norm(theta)

    plt.plot(lmbdas, mses, "o-", label = "MSE score")
    plt.xlabel("$\\lambda$")
    plt.ylabel("MSE")
    plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()
    plt.plot(lmbdas, R2s, "o-", label = "$R^2$ score")
    plt.xlabel("$\\lambda $")
    plt.ylabel("$R^2$")
    plt.xscale("log")
    plt.legend()
    plt.show() 
    plt.plot(lmbdas, theta_norms, "o-", label = "$\\|\\theta \\|$")
    plt.xlabel("$\\lambda$")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rs = 2026
    x, y = runge_data()

    pow = 25
    ts = 0.2
    plot_score_lmbdas(x, y, pow, ts, rs)