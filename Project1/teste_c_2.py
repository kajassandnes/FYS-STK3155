from part_a import *
from sklearn.utils import resample
np.random.seed(2026)
n, n_bootstraps, maxdegree = 40, 100, 14

x = np.linspace(-1, 1, n).reshape(-1, 1)
y = 1 / (1 + (25 * x**2)) + np.random.normal(0, 0.1, x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=2026)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
for degree in range(1, maxdegree):
    X_test = design_matrix(x_test, degree)
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_ = resample(x_train, y_train)
        X_ = design_matrix(x_, degree)
        theta = OLS(X_, y_)

        y_pred[:, i] = (X_test @ theta).ravel()

    # Expectations over training sets are averages along axis 1;
    # keepdims=True preserves the column shape and is essential
    # for the bias to come out right.
    error[degree - 1] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias[degree - 1] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
    variance[degree - 1] = np.mean(np.var(y_pred, axis=1, keepdims=True))

fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(range(maxdegree), error, "o-", label="test error")
ax.plot(range(maxdegree), bias, "s-", label=r"bias$^2$ (+ $\sigma^2$)")
ax.plot(range(maxdegree), variance, "d-",label="variance")
ax.set_yscale("log")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("MSE decomposition")
ax.legend(frameon=False)
plt.show()
