from part_f import *
import sklearn.linear_model as skl


def cost_lasso(theta, X, y, lmda):
    return jnp.mean((y - X @ theta)**2) + lmda * jnp.sum(jnp.abs(theta))

_grad_cost_lasso = jax.jit(jax.grad(cost_lasso))


def gradient_lasso_analytical(theta, X, y, lmda):
    """Returns analyical gradient of the costfunction.
    lmda = 0 reduces to the OLS-gradient
    """
    n = len(y)
    gradient = 2/n * X.T @ (X @ theta - y) + lmda*np.sign(theta)
    return gradient


def gradient_lasso_automatic_diff(theta, X, y, lmda):
    """Return gradient calculated by automatic differentiation"""
    return np.asarray(_grad_cost_lasso(theta, X, y, lmda)) 


def gradient_lasso_smooth(theta, X, y, lmda):
    n = X.shape[0]
    return 2/n * X.T @ (X @ theta - y)

def soft_threshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def proximal_gradient_descent(theta, state, eta, grad, lmda):
    x = theta - eta*grad
    alpha = eta*lmda
    theta = soft_threshold(x, alpha)

    return theta, state
    


def theta_lasso_sklearn(X, y, lmda):
    """sklearn's Lasso method minimizes (1 / (2 * n)) * ||y - X theta||^2_2 + alpha * ||theta||_1
    Therefore we use alpha = lmda/2
    """
    alpha = lmda/2
    model = skl.Lasso(alpha=alpha, fit_intercept=False, max_iter=100000)
    model.fit(X, y)

    return model.coef_


def plot_lasso_difference(x, y, degree, gradient_method, descent_method, exact_method, etas, lmdas, rs, ts):
    fig, ax = plt.subplots()
    for lmda in lmdas:
        iterations = []
        differences = []
        for eta in etas:
            iteration, difference = theta_numeric(x, y, degree, gradient_method, descent_method, exact_method, lmda, eta, rs, ts)
            iterations.append(iteration)
            differences.append(difference)
        ax.plot(etas, differences, 'o-', label=rf'$\lambda={lmda}$')

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$|\hat{\theta} - \theta|$')
    ax.set_yscale('log')
    ax.set_title('Difference in Lasso Computed by Gradient vs Coordinate Descent')
    ax.legend()
    plt.show()


def gradient_at_zero():
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    X = design_matrix(x, degree)
    lmda = 0.1

    theta_vals = np.array([-0.1, -0.01, 0, 0.01, 0.1])
    norms_analytical = np.zeros_like(theta_vals)
    norms_automatic = np.zeros_like(theta_vals)

    i = 0
    for val in theta_vals:
        theta = np.ones(degree)*val
        ana_grad = gradient_lasso_analytical(theta, X, y, lmda)
        auto_grad = gradient_lasso_automatic_diff(theta, X, y, lmda)

        norms_analytical[i] = np.linalg.norm(ana_grad)
        norms_automatic[i] = np.linalg.norm(auto_grad)

        i += 1


    plt.plot(theta_vals, norms_analytical, 'o', markersize=8, label="analytical")
    plt.plot(theta_vals, norms_automatic, 'o', markersize=5, label="automatic")
    plt.xlabel(r'$\theta$-parameter value')
    plt.ylabel(r'$|\nabla\theta|$')
    plt.title("Gradient Calculated Analytic vs Automatic")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_lasso_smooth
    descent_method = proximal_gradient_descent
    exact_method = theta_lasso_sklearn
    ts = 0.2
    tol = 1e-2

    etas = np.linspace(0.02, 0.35, 8)
    lmdas = [0.01, 0.05, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]

    plot_eta_lambda(x, y, degree, gradient_method, descent_method, exact_method, etas, lmdas, rs, ts, tol)
    #plot_lasso_difference(x, y, degree, gradient_method, descent_method, exact_method, etas, lmdas, rs, ts)
    gradient_at_zero()


    print("jax.grad(jnp.abs)(0.0) =", jax.grad(jnp.abs)(0.0))
    print("jax.grad(jnp.abs)(-0.3) =", jax.grad(jnp.abs)(-0.3), " jax.grad(jnp.abs)(0.3) =", jax.grad(jnp.abs)(0.3))