from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from part_a import *
import warnings
from sklearn.exceptions import ConvergenceWarning


# DENNE FUNKSJONEN ER AI-GENERERT
def kfold(x, y, k, p, model, rs):
    """
    Return mean and standard error of K-fold validation MSE.

    Returns (np.nan, np.nan) if Lasso does not converge in one or
    more folds.
    """

    pipe = make_pipeline(
        PolynomialFeatures(degree=p, include_bias=False),
        StandardScaler(),
        model
    )

    cv = KFold(n_splits=k, shuffle=True, random_state=rs)

    # Record convergence warnings instead of printing many warnings.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)

        scores = cross_val_score(
            pipe,
            x[:, np.newaxis],
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            error_score=np.nan
        )

    # Treat non-converged Lasso fits as invalid parameter combinations.
    has_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught_warnings
    )

    mses = -scores

    if has_convergence_warning or np.any(np.isnan(mses)):
        return np.nan, np.nan

    mean_mse = np.mean(mses)
    se_mse = np.std(mses, ddof=1) / np.sqrt(k)

    return mean_mse, se_mse

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

# DENNE FUNKSJONEN ER AI-GENERERT
def plot_cv_heatmap(x, y, k, p_range, lmbda_range, model_class, rs):
    """
    Plot K-fold CV MSE as a function of polynomial degree and lambda.

    Parameters
    ----------
    model_class : Ridge or Lasso
        The sklearn estimator class, not an instantiated model.
    """

    if model_class == Ridge:
        model_name = "Ridge"

        # Ridge has a closed-form-type stable solver by default.
        model_kwargs = {
            "fit_intercept": True
        }

    elif model_class == Lasso:
        model_name = "Lasso"

        # Raise max_iter substantially for high-degree polynomial features.
        model_kwargs = {
            "fit_intercept": True,
            "max_iter": 100_000,
            "tol": 1e-6,
            "selection": "cyclic"
        }

    else:
        raise ValueError("model_class must be Ridge or Lasso.")

    # Rows correspond to lambda values; columns correspond to degrees.
    mean_mses = np.full((len(lmbda_range), len(p_range)), np.nan)
    se_mses = np.full((len(lmbda_range), len(p_range)), np.nan)

    for i, lmbda in enumerate(lmbda_range):
        for j, p in enumerate(p_range):

            model = model_class(alpha=lmbda, **model_kwargs)

            mean_mses[i, j], se_mses[i, j] = kfold(
                x=x,
                y=y,
                k=k,
                p=p,
                model=model,
                rs=rs
            )

    # Find the best converged combination.
    if np.all(np.isnan(mean_mses)):
        raise RuntimeError(
            f"No {model_name} models converged for the specified grid."
        )

    best_i, best_j = np.unravel_index(
        np.nanargmin(mean_mses),
        mean_mses.shape
    )

    best_lambda = lmbda_range[best_i]
    best_degree = p_range[best_j]
    best_mse = mean_mses[best_i, best_j]
    best_se = se_mses[best_i, best_j]

    print(
        f"{model_name} minimum:\n"
        f"  degree = {best_degree}\n"
        f"  lambda = {best_lambda:.3e}\n"
        f"  CV MSE = {best_mse:.6g} ± {best_se:.3g}"
    )

    # Mask NaNs, e.g. failed/non-converged Lasso combinations.
    masked_mses = np.ma.masked_invalid(mean_mses)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    mesh = ax.pcolormesh(
        p_range,
        lmbda_range,
        masked_mses,
        shading="nearest",
        cmap=cmap
    )

    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel(r"Regularization parameter $\lambda$")
    ax.set_title(f"{k}-fold CV MSE: {model_name}")

    # Mark the best model.
    ax.plot(
        best_degree,
        best_lambda,
        marker="*",
        markersize=16,
        markeredgecolor="black",
        markerfacecolor="red",
        label=(
            f"Best: $p={best_degree}$, "
            rf"$\lambda={best_lambda:.1e}$"
        )
    )

    ax.legend(loc="best")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Mean cross-validated MSE")

    plt.tight_layout()
    plt.show()

    return mean_mses, se_mses, best_degree, best_lambda


if __name__ == "__main__":

    
    nlmbdas = 20
    lmbda_range_ridge = np.logspace(-10, 2, nlmbdas)
    lmbda_range_lasso = np.logspace(-10, 2, nlmbdas)
    k = 5
    min_degree = 1
    max_degree = 25
    pow_range = np.arange(min_degree, max_degree + 1)
    rs = 3155
    x, y = runge_data(n = 100, std = 0.1)

    ols = LinearRegression(fit_intercept = True)
    plot_kfold_powers(x, y, k, max_degree, ols, rs)
    #plot_kfold_lmbdas_powers(x, y, k, pow_range, lmbda_range_ridge, Ridge, rs)
    #plot_kfold_lmbdas_powers(x, y, k, pow_range, lmbda_range_lasso, Lasso, rs)

    # DISSE FUNKSJONSKALLENE ER AI-GENERERTE
    '''
    ridge_mses, ridge_ses, ridge_degree, ridge_lambda = plot_cv_heatmap(
        x=x,
        y=y,
        k=k,
        p_range=pow_range,
        lmbda_range=lmbda_range_ridge,
        model_class=Ridge,
        rs=rs
    )
    lasso_mses, lasso_ses, lasso_degree, lasso_lambda = plot_cv_heatmap(
        x=x,
        y=y,
        k=k,
        p_range=pow_range,
        lmbda_range=lmbda_range_lasso,
        model_class=Lasso,
        rs=rs
    )
    '''