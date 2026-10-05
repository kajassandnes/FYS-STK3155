# this program use gradient descendent to find the optimal fit for Runge's function
from part_a import *
import jax
import jax.numpy as jnp
from jax import grad, jit
jax.config.update("jax_enable_x64", True)   # 64-bit floats, as in numpy

# parameters for plotting
plt.rcParams['axes.titlesize'] = 24   # Axis title
plt.rcParams['axes.labelsize'] = 20    # X and Y axis labels (names)
plt.rcParams['xtick.labelsize'] = 18   # X axis values (ticks)
plt.rcParams['ytick.labelsize'] = 18   # Y axis values (ticks)
plt.rcParams['legend.fontsize'] = 16   # Legend text
plt.rcParams['figure.titlesize'] = 26  # Overall figure title (suptitle)

def check_gradients(X, y, lmda, rs, eta, tol=1e-15, plain_gradient_descent=False):
    """Compares analytical gradient against the autodiff gradient for OLS and Ridge.
    """
    rng = np.random.default_rng(rs)
    theta = rng.normal(size=X.shape[1])

    g_analytical = gradient_analytical(theta, X, y, lmda)
    g_automatic = gradient_automatic_diff(theta, X, y, lmda)

    diff = np.max(np.abs(g_analytical - g_automatic))
    assert diff < tol, f"gradient mismatch at lmda={lmda}: max|diff|={diff}, tolerance={tol}, degree={X.shape[1]}"

    print(f"\nAnalytical and autodiff gradients for OLS/Ridge agree to within {tol:.0e}")

    eta_bound = eta_max(X, lmda)
    if eta >= eta_bound and plain_gradient_descent:
        print(f"WARNING: eta={eta} exceeds the theoretical bound eta_max={eta_bound:.4g}; "
              f"gradient descent may diverge.")


def eta_max(X, lmda):
    """Returns largest theoretical learning rate using the Hessian.
    Valid for plain gradient descent.
    """
    n = X.shape[0]
    H = 2.0/n * X.T @ X + 2.0*lmda*np.eye(X.shape[1])
    eigvals = np.linalg.eigvalsh(H)
    print("Condition number:", eigvals.max() / eigvals.min())
    return 2.0 / np.linalg.eigvalsh(H).max()


def gradient_analytical(theta, X, y, lmda):
    """Returns analyical gradient of the Ridge cost function.
    lmda = 0 equals the OLS-gradient
    """
    n = len(y)
    gradient = 2/n * X.T @ (X @ theta - y) + 2*lmda/n*theta
    return gradient


def gradient_automatic_diff(theta, X, y, lmda):
    """Return gradient of Ridge cost function calculated by 
    automatic differentiation. lmda = 0 equals the OLS-gradient
    """
    return np.asarray(_grad_cost(theta, X, y, lmda))    


def cost(theta, X, y, lmda):
    """Cost function for Ridge
    lmda = 0 equals the cost function of OLS
    """
    n = X.shape[0]
    return 1/n * jnp.linalg.norm(X @ theta - y)**2 + lmda / n * jnp.linalg.norm(theta)**2

_grad_cost = jax.jit(jax.grad(cost))


def gradient_descent(X, y, theta, grad_meth, lmda, eta, theta_exact, tol=1.0e-8):
    """Returns theta calculated by plain gradient descent, the number of itterations
    and the running differences between theta and the closed form solution.

    The methods available to calculate the gradient is: OLS, Ridge, automatic differentiation
    """
    history = {'theta_diff': []}
    for k in range(10000):
        old_theta = theta
        gradient = grad_meth(theta, X, y, lmda)
        theta = theta - eta * gradient

        history["theta_diff"].append(float(np.linalg.norm(theta - theta_exact)))

        if np.linalg.norm(theta - old_theta) < tol:
            break

    return theta, k+1, history


def theta_numeric(data, target, degree, gradient_method, lmda, eta, rs, ts):
    """Input: observed data, target, maximum polynomial degree, 
    method of which to calculate gradient, penalty scalar, learning rate, 
    random seed and test size
    
    Splits data into training and test data, calculates theta by gradient descent.
    Prints difference between numerical and exact theta.
    """
    X = design_matrix(data, degree)
    X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

    # check of gradients and learning rate
    check_gradients(X_train_scaled, y_train_centered, lmda, rs, eta, plain_gradient_descent=True)

    # calculate exact theta
    n = len(y_train_centered)
    theta_exact = np.linalg.pinv(X_train_scaled.T @ X_train_scaled + lmda*np.eye(X_train_scaled.shape[1])) @ X_train_scaled.T @ y_train_centered

    # calculate numerical theta
    rng = np.random.default_rng(rs)
    theta0 = rng.normal(size=(degree))
    theta, iterations, history = gradient_descent(X_train_scaled, y_train_centered, theta0, gradient_method, lmda, eta, 
                                                  theta_exact)

    # compares analytical and numerical theta
    print(f"analytical theta:                               {theta_exact.ravel()}")
    print(f"Difference between exact and numerical theta:   {theta_exact - theta}")

    return iterations, history

def plot_eta_lambda(x, y, degree, gradient_method, etas, lmdas, rs, ts, target_accuracy=1e-4):
    """Inputs: dataset, target, maximum polynomial degree, method to compute gradient, 
    learning rates, penalty parameters, random seed, test size. Default target accuracy 
    describes the criteria of convergense to closed form solution.

    The function plots iterations to convergence and closed form solution vs learning rate
    for different values of the penalty parameter.
    
    Partially written with help of AI, which helped with the set up  of loops and finding
    the iteration when theta converged to closed form solution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
    
    for lmda in lmdas:
        iterations = []
        iters_to_target = []

        for eta in etas:
            iteration, history = theta_numeric(x, y, degree, gradient_method, lmda, eta, rs, ts)
            iterations.append(iteration)
        
        # find first iteration where theta got within target_accuracy of theta_exact
            diffs = np.array(history["theta_diff"])
            below = np.where(diffs < target_accuracy)[0]
            iters_to_target.append(below[0]+1 if len(below) > 0 else np.nan)

        # plotting iterations to converge
        ax1.plot(etas, iterations, 'o-', label=rf'$\lambda={lmda}$')

        # plotting iterations to converge to closed solution
        ax2.plot(etas, iters_to_target, 'o-', label=rf'$\lambda={lmda}$')

    # iterations to converge
    ax1.set_ylabel(r'iterations')
    ax1.set_yscale('log')
    ax1.set_title(r'Convergence Speed')
    ax1.legend()

    # iterations to converge to closed solution
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel(rf'iterations')
    ax2.set_yscale('log')
    ax2.set_title('Convergence to Closed-Form Solution')
    ax2.legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_automatic_diff  
    ts = 0.2

    etas = np.linspace(0.01, 0.4, 10)
    lmdas = [0.0, 0.01, 0.1, 1.0]

    plot_eta_lambda(x, y, degree, gradient_method, etas, lmdas, rs, ts)