from part_f import *
import sklearn.linear_model as skl


def cost_lasso(theta, X, y, alpha):
    """Cost function for Lasso"""
    n = X.shape[0]
    return 1/(2*n) * jnp.linalg.norm(y - X @ theta)**2 + alpha * jnp.sum(jnp.abs(theta))

_grad_cost_lasso = jax.jit(jax.grad(cost_lasso))


def gradient_lasso_analytical(theta, X, y, lmda):
    """Returns analyical gradient of the cost function.
    lmda = 0 reduces to the OLS-gradient
    """
    n = len(y)
    gradient = 2/n * X.T @ (X @ theta - y) + lmda*np.sign(theta)
    return gradient


def gradient_lasso_automatic_diff(theta, X, y, lmda):
    """Returns gradient calculated by automatic differentiation
    """
    return np.asarray(_grad_cost_lasso(theta, X, y, lmda)) 


def gradient_lasso_smooth(theta, X, y, lmda):
    """Returns the gradient of the smooth part of the 
    lasso cost function, which is the OLS gradient.
    """
    n = X.shape[0]
    return 2/n * X.T @ (X @ theta - y)


def soft_threshold(x, alpha):
    """The function behaves as the soft thresholding operatior, 
    which returns 0 if abs|x| < alpha.
    """
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)


def proximal_gradient_descent(theta, state, eta, grad, lmda):
    """Returns theta after update in the direction of the
    negative gradient, then use of the soft threshold operator
    """
    x = theta - eta*grad
    alpha = eta*lmda
    theta = soft_threshold(x, alpha)

    return theta, state
    

def theta_lasso_sklearn(X, y, lmda):
    """Returns theta minimized by sklearn's Lasso coordinate descent.
    """
    model = skl.Lasso(alpha=lmda, fit_intercept=False, max_iter=10000)
    model.fit(X, y)

    return model.coef_


def gradient_at_zero():
    """Plots the norm of the analytical gradient and the gradient 
    computed by automatic differentiation of the lasso cost function
    around and at theta=0.
    """
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
    lmdas = np.arange(0.1, 1, 0.1)

    #plot_eta_lambda(x, y, degree, gradient_method, descent_method, exact_method, lmdas, etas, rs, ts)
    plot_lasso_difference(x, y, degree, gradient_method, descent_method, exact_method, lmdas, etas, rs, ts)
    #gradient_at_zero()


    print("jax.grad(jnp.abs)(0.0) =", jax.grad(jnp.abs)(0.0))
    print("jax.grad(jnp.abs)(-0.3) =", jax.grad(jnp.abs)(-0.3), " jax.grad(jnp.abs)(0.3) =", jax.grad(jnp.abs)(0.3))