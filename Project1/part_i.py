from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from part_a import *

def kfold(x, y, k, p, model, rs):

    pipe = make_pipeline(PolynomialFeatures(degree = p, include_bias = False), StandardScaler(), model)
    kfold = KFold(n_splits = k, shuffle = True, random_state = rs)
    scores = cross_val_score(pipe, x[:, np.newaxis], y, cv = kfold, scoring = "neg_mean_squared_error")

    return np.mean(-scores), np.std(-scores)

def plot_kfold_powers(x, y, k, degree_max, model, rs):

    powers = np.arange(1, degree_max + 1)
    mses = np.zeros(degree_max)
    stds = np.zeros(degree_max)
    for p in powers:
        mses[p - 1], stds[p - 1] = kfold(x, y, k, p, model, rs)
        stds[p - 1] = stds[p - 1] / np.sqrt(k)

    mindex = np.argmin(mses)
    print(f"Minimizing degree: {mindex + 1}, minimized error: {mses[mindex]} +- {stds[mindex]}")
    plt.plot(powers, mses, "o-", label = "$MSE$")
    plt.xlabel("Powers")
    plt.title(f"{k}-fold cross validation: OLS on runge data")
    plt.legend()
    plt.show()

def plot_kfold_lmbdas(x, y, k, p, lmbda_range, model, rs):

    mses = np.zeros_like(lmbda_range)
    stds = np.zeros_like(lmbda_range)
    for i in range(len(lmbda_range)):
        mses[i], stds[i] = kfold(x, y, k, p, model(fit_intercept = True, alpha = lmbda_range[i]), rs)
        stds[i] = stds[i] / np.sqrt(k)

    plt.plot(lmbda_range, mses, "o-", label = f"$Degree = {p}$")
    return mses, stds

def plot_kfold_lmbdas_powers(x, y, k, p_range, lmbda_range, model, rs):

    if model == Ridge:
        model_name = "Ridge"
    elif model == Lasso:
        model_name = "Lasso"
    else:
        raise ValueError("Model must be either sklearn Ridge og Lasso")
    minmses = np.zeros_like(p_range, dtype = float)
    minstds = np.zeros_like(p_range, dtype = float)
    minlmbdas = np.zeros_like(p_range, dtype = float)

    for p in range(len(p_range)):

        mses, stds = plot_kfold_lmbdas(x, y, k, p_range[p], lmbda_range, model, rs)
        mindex = np.argmin(mses)
        minmses[p] = mses[mindex]
        minstds[p] = stds[mindex]
        minlmbdas[p] = lmbda_range[mindex]

    mindex = np.argmin(minmses)
    print(f"Minimizing degree: {p_range[mindex]}, minimizing lambda: {minlmbdas[mindex]}, minimized error: {minmses[mindex]} +- {minstds[mindex]}")

    plt.xlabel("$\\lambda$")
    plt.title(f"MSE {k}-fold cross validation: {model_name} on runge data.")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    model = LinearRegression(fit_intercept = True)
    nlmbdas = 20
    lmbda_range_ridge = np.logspace(-10, 2, nlmbdas)
    lmbda_range_lasso = np.logspace(-6, 2, nlmbdas)
    k = 5
    min_degree = 1
    max_degree = 25
    pow_range = np.arange(min_degree, max_degree + 1)
    rs = 3155
    x, y = runge_data(n = 100, std = 0.1)

    plot_kfold_powers(x, y, k, max_degree, model, rs)
    #plot_kfold_lmbdas_powers(x, y, k, pow_range, lmbda_range_ridge, Ridge, rs)
    #plot_kfold_lmbdas_powers(x, y, k, pow_range, lmbda_range_lasso, Lasso, rs)