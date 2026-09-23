from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from part_a import *

def kfold(x, y, k, p, model, rs):

    pipe = make_pipeline(PolynomialFeatures(degree = p, include_bias = False), StandardScaler(), model)
    kfold = KFold(n_splits = k, shuffle = True, random_state = rs)
    scores = cross_val_score(pipe, x[:, np.newaxis], y, cv = kfold, scoring = "neg_mean_squared_error")

    return np.mean(-scores)

def plot_kfold_powers(x, y, k, degree_max, model, rs):

    powers = np.arange(1, degree_max + 1)
    mses = np.zeros(degree_max)
    for p in powers:
        mses[p - 1] = kfold(x, y, k, p, model, rs)

    plt.plot(powers, mses, "o-", label = "$MSE$")
    plt.xlabel("Powers")
    plt.title(f"{k}-fold cross validation: OLS on runge data")
    plt.legend()
    plt.show()

def plot_kfold_lmbdas(x, y, k, p, lmbda_range, rs):

    mses = np.zeros_like(lmbda_range)
    for i in range(len(lmbda_range)):
        mses[i] = kfold(x, y, k, p, Ridge(fit_intercept = True, alpha = lmbda_range[i]), rs)

    plt.plot(lmbda_range, mses, "o-", label = f"$Degree = {p}$")

def plot_kfold_lmbdas_powers(x, y, k, p_range, lmbda_range, rs):

    for p in p_range:
        plot_kfold_lmbdas(x, y, k, p, lmbda_range, rs)

    plt.xlabel("$\\lambda$")
    plt.title(f"MSE {k}-fold cross validation: Ridge on runge data.")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    model = LinearRegression(fit_intercept = True)
    nlmbdas = 20
    lmbda_range = np.logspace(-8, 2, nlmbdas)
    pow_range = np.arange(15, 21)
    p = 20
    k = 5
    max_degree = 25
    rs = 2026
    x, y = runge_data(n = 100)

    plot_kfold_powers(x, y, k, max_degree, model, rs)
    #plot_kfold_lmbdas_powers(x, y, k, pow_range, lmbda_range, rs)