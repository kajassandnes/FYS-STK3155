from part_a import *


def ridge(X, target, lmbda = 0.1):
    """
    Input data, lmbda, and target variable. return polynomial coefficients.
    Uses ridge regression with np.linalg.solve().
    """
    n, p = X.shape
    I = np.eye(p)

    theta = np.linalg.solve(X.T @ X + lmbda * I, X.T @ target)
    return theta


def plot_score_lmbdas(data, target, pow,  ts, rs, lmbda_min = -8, lmbda_max = 2, nlmbdas = 20, plot_rest = False):
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

    if plot_rest == False:
        plt.plot(lmbdas, mses, ".-", label = f"Degree = {pow}")

    if plot_rest == True:
        plt.plot(lmbdas, mses, ".-", label = f"Degree = {pow}")
        plt.xlabel("$\\lambda$")
        plt.ylabel("MSE")
        plt.title("MSE for ridge model")
        plt.xscale("log")
        plt.legend()
        plt.show()
        plt.plot(lmbdas, R2s, "o-", label = f"Degree = {pow}")
        plt.xlabel("$\\lambda $")
        plt.ylabel("$R^2$")
        plt.title("$R^2$-score for ridge model")
        plt.xscale("log")
        plt.legend()
        plt.show() 
        plt.plot(lmbdas, theta_norms, "o-", label = f"Degree = {pow}")
        plt.xlabel("$\\lambda$")
        plt.ylabel("$\\|\\theta \\|$")
        plt.title("Ridge parameter shrinkage")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.show()


def plot_ridge_powers(data, target, pow_min, pow_max, ts, rs, lmbda_min = -8, lmbda_max = 2, nlmbdas = 20):

    powers = np.arange(pow_min, pow_max + 1)
    for pow in powers:
        plot_score_lmbdas(data, target, pow, ts, rs, lmbda_min, lmbda_max, nlmbdas)

    ols_min = 0.0067931
    ols_vals = np.ones(nlmbdas) * ols_min
    lmbdas = np.logspace(lmbda_min, lmbda_max, nlmbdas)
    plt.plot(lmbdas, ols_vals, "--", label = "Minimum MSE for OLS")
    plt.xlabel("$\\lambda$")
    plt.ylabel("MSE")
    plt.title("MSE for ridge models")
    plt.xscale("log")
    plt.legend()
    plt.show()

def test_ridge(tol = 1e-9):
    x, y = runge_data()
    X = design_matrix(x, 6)
    theta_ols = OLS(X, y)
    theta_ridge = ridge(X, y, lmbda = 0.0)

    assert np.max(theta_ols - theta_ridge) < tol

if __name__ == "__main__":
    test_ridge()
    rs = 2026
    x, y = runge_data()

    pow_min = 12
    pow_max = 25
    ts = 0.2
    plot_ridge_powers(x, y, pow_min, pow_max, ts, rs)
    plot_score_lmbdas(x, y, pow_max, ts, rs, plot_rest = True)