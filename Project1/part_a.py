import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def design_matrix(data, degree):
    """
    Input data and degree of polynomial. Return design matrix
    """
    X = np.column_stack([data ** j for j in range(1, degree + 1)])
    return X

def split_scale(X, target, ts, rs):
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = ts, random_state = rs)
    
    # making sure to avoid data leakage:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_centered = y_train - np.mean(y_train)
    y_test_centered = y_test - np.mean(y_train)

    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered

def runge_data(n = 100, std = 0.1, rs = 2026, x0 = -1, x1 = 1):
    rng = np.random.default_rng(rs)

    x = np.linspace(x0, x1, n)
    y = (1 / (1 + (25 * (x ** 2)))) + (std * rng.standard_normal(len(x)))

    return x, y

def OLS(dsgn_mtrx, target):
    """
    Input data and target variable. Return the polynomial coefficients.
    Uses Moore-Penrose pseudoinverse.
    """
    theta = np.linalg.pinv(dsgn_mtrx) @ target
    return theta

def plot_score_powers(data, target, ts, degree_max, method, rs):
    """
    Input data, target variable, training size, and maximum degree of polnomial.
    Plots mse, r2, and coefficients.
    """
    powers = np.arange(1, degree_max + 1)
    theta_norms = np.zeros(degree_max)
    mses = np.zeros_like(powers, dtype = float)
    R2s = np.zeros_like(powers, dtype = float)

    for p in range(1, degree_max + 1):
        X = design_matrix(data, p)
        X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

        theta = method(X_train_scaled, y_train_centered)
        y_tilde = X_test_scaled @ theta
        mse_score, R2_score = mse(y_test_centered, y_tilde), r2_score(y_test_centered, y_tilde)

        mses[p - 1] = mse_score
        R2s[p - 1] = R2_score
        theta_norms[p - 1] = np.linalg.norm(theta)


    plt.plot(powers, mses, "o-", label = "MSE score")
    plt.xlabel("Powers")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    plt.plot(powers, R2s, "o-", label = "$R^2$ score")
    plt.xlabel("Powers")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    plt.plot(powers, theta_norms, "o-", label = "$\\|\\theta \\|$")
    plt.xlabel("Powers")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    x, y = runge_data()
    plot_score_powers(x, y, ts = 0.2, degree_max = 15, method = OLS, rs = 2026)