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