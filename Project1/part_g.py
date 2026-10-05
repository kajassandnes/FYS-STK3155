from part_f import *
import sklearn.linear_model as skl


def cost_lasso(theta, X, y, alpha):
    """Cost function for Lasso"""
    n = X.shape[0]
    return 1/(2*n) * jnp.linalg.norm(y - X @ theta)**2 + alpha * jnp.sum(jnp.abs(theta))

_grad_cost_lasso = jax.jit(jax.grad(cost_lasso))


def gradient_lasso_analytical(theta, X, y, lmda):
    """Returns analyical subgradient of the Lasso cost.
    Matches sklearn: (1/(2n))||y - X theta||^2 + lmda ||theta||_1.
    lmda = 0 reduces to the OLS-gradient of that quadratic term.
    """
    n = len(y)
    gradient = 1/n * X.T @ (X @ theta - y) + lmda * np.sign(theta)
    return gradient


def gradient_lasso_automatic_diff(theta, X, y, lmda):
    """Returns gradient calculated by automatic differentiation
    """
    return np.asarray(_grad_cost_lasso(theta, X, y, lmda)) 


def gradient_lasso_smooth(theta, X, y, lmda):
    """Returns the gradient of the smooth part of the Lasso cost,
    (1/(2n))||y - X theta||^2. The L1 term is handled by the proximal step.
    """
    n = X.shape[0]
    return 1/n * X.T @ (X @ theta - y)


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
    

def plain_gd(theta, state, eta, grad, *args, **kwargs):
    """Plain gradient descent from part e, as a descent_method."""
    return theta - eta * grad, state


def theta_lasso_sklearn(X, y, lmda):
    """Returns theta minimized by sklearn's Lasso coordinate descent.
    """
    model = skl.Lasso(alpha=lmda, fit_intercept=False, max_iter=10000)
    model.fit(X, y)

    return model.coef_


def gradient_at_zero():
    """Plots the norm of the analytical subgradient and the autodiff gradient
    of the Lasso cost around and at theta=0.

    jnp.abs is not differentiable at 0. JAX returns 0 there, which is a valid
    subgradient of |x| (the subdifferential is [-1, 1]), but it is only one
    choice. Using that value in ordinary GD does not enforce sparsity; the
    proximal / soft-threshold step does.
    """
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    X = design_matrix(x, degree)
    lmda = 0.1

    print("jax.grad(jnp.abs)(0.0) =", float(jax.grad(jnp.abs)(0.0)))
    print("jax.grad(jnp.abs)(-0.3) =", float(jax.grad(jnp.abs)(-0.3)))
    print("jax.grad(jnp.abs)(0.3)  =", float(jax.grad(jnp.abs)(0.3)))
    print("Subdifferential of |x| at 0 is [-1, 1]; JAX's 0 is valid but not the only subgradient.")

    theta_vals = np.array([-0.1, -0.01, 0, 0.01, 0.1])
    norms_analytical = np.zeros_like(theta_vals)
    norms_automatic = np.zeros_like(theta_vals)

    for i, val in enumerate(theta_vals):
        theta = np.ones(degree) * val
        norms_analytical[i] = np.linalg.norm(gradient_lasso_analytical(theta, X, y, lmda))
        norms_automatic[i] = np.linalg.norm(gradient_lasso_automatic_diff(theta, X, y, lmda))

    plt.plot(theta_vals, norms_analytical, "o", markersize=8, label="analytical sign")
    plt.plot(theta_vals, norms_automatic, "o", markersize=5, label="autodiff")
    plt.xlabel(r"$\theta$-parameter value")
    plt.ylabel(r"$|\nabla C|$")
    plt.title("Lasso Gradient: Analytic vs Automatic")
    plt.legend()
    plt.show()


def plot_lasso_methods(x, y, degree, methods, exact_method, lmdas, eta, rs, ts, tol=1e-8, target_accuracy=1e-4):
    """Plot Lasso iterations vs lambda for Proximal GD, Momentum, AdaGrad, RMSprop and Adam.

    sklearn is only used through exact_method (theta_lasso_sklearn) as the reference solution.

    LLM-assisted
    ------------
    Tool: Cursor (October 2026)
    Role: The LLM wrote the entire function inspired by a previous draft written by myself.
    """
    fig, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    ax1, ax2 = axs

    for gradient_method, descent_method, label, color in methods:
        iterations = []
        iters_to_target = []
        for lmda in lmdas:
            iteration, history = theta_numeric(
                x, y, degree, gradient_method, descent_method, exact_method,
                lmda, eta, rs, ts, tol, check=False
            )
            iterations.append(iteration)

            diffs = np.array(history["theta_diff"])
            below = np.where(diffs < target_accuracy)[0]
            iters_to_target.append(below[0] + 1 if len(below) > 0 else np.nan)

        ax1.plot(lmdas, iterations, "o-", color=color, label=label)
        ax2.plot(lmdas, iters_to_target, "o-", color=color, label=label)

    ax1.set_ylabel(r"Iterations")
    #ax1.set_yscale("log")
    ax1.set_title(rf"Lasso Convergence Speed ($\eta_0={eta}$)")
    ax1.legend()

    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"Iterations")
    #ax2.set_yscale("log")
    ax2.set_title(rf"Convergence to sklearn Lasso ($\eta_0={eta}$)")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    exact_method = theta_lasso_sklearn
    ts = 0.2
    tol=1e-8
    eta = 0.01
    lmdas = np.arange(0.1, 1, 0.1)

    methods = [
        (gradient_lasso_smooth, proximal_gradient_descent, "Proximal GD", "black"),
        (gradient_lasso_analytical, momentum, "Momentum", "blue"),
        (gradient_lasso_analytical, AdaGrad, "AdaGrad", "orange"),
        (gradient_lasso_analytical, RMSprop, "RMSprop", "green"),
        (gradient_lasso_analytical, Adam, "Adam", "red"),
    ]

    plot_lasso_methods(x, y, degree, methods, exact_method, lmdas, eta, rs, ts, tol)
    gradient_at_zero()
