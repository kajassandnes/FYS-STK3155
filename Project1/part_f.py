
from part_e import *


def momentum(theta, state, eta, grad, mom_par=0.9, *args, **kwargs):
    """Gradient descent using momentum method
    """
    change = eta*grad + mom_par*state.get('v', np.zeros_like(theta))
    theta = theta - change
    state = {'v': change}

    return theta, state


def AdaGrad(theta, state, eta, grad, eps=1e-8, *args, **kwargs):
    """Gradient descent using AdaGrad method
    """
    r = state.get('r', np.zeros_like(theta)) + grad**2
    theta = theta - eta * grad / (np.sqrt(r) + eps) 
    state = {'r': r}

    return theta, state


def RMSprop(theta, state, eta, grad, rho=0.999, eps=1e-8, *args, **kwargs):
    """Gradient descent using RMSprop method
    """
    v = rho * state.get('v', np.zeros_like(theta)) + (1 - rho) * grad**2
    theta = theta - eta * grad / (np.sqrt(v) + eps)
    state = {'v': v}

    return theta, state


def Adam(theta, state, eta, grad, beta1=0.9, beta2=0.999, eps=1e-8, *args, **kwargs):
    """Gradient descent using Adam method
    """
    t = state.get('t', 0) + 1
    m = (beta1 * state.get('m', np.zeros_like(theta)) + (1-beta1) * grad) 
    v = (beta2 * state.get('v', np.zeros_like(theta)) + (1-beta2) * grad**2) 
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)
    state = {'m': m, 'v': v, 't': t}

    return theta, state


def ols_ridge_exact(X, y, lmda):
    """Return closed form solution of Ridge. 
    If lmda is 0, solution reduces to OLS.
    """
    n = len(y)
    return np.linalg.pinv(X.T @ X + lmda*np.eye(X.shape[1])) @ X.T @ y


def gradient_descent_general(X, y, theta, gradient_method, descent_method, lmda, eta, theta_exact, tol):
    """This is a generalized function of "gradient descent" in part_e.

    Returns theta calculated by plain gradient descent, the number of itterations
    and the running differences between theta and the closed form solution.

    The methods available to calculate the gradient is: OLS, Ridge, automatic differentiation
    The gradient descent methods available is: momentum, AdaGrad, RMSprop, Adam
    """
    state = {}
    history = {'theta_diff': []}
    for k in range(10000):
        old_theta = theta
        gradient = gradient_method(theta, X, y, lmda)
        theta, state = descent_method(theta, state, eta, gradient, lmda=lmda)

        history["theta_diff"].append(float(np.linalg.norm(theta - theta_exact)))

        # cheks if theta has converged
        if np.linalg.norm(theta - old_theta) < tol:
            break

    return theta, k+1, history


def theta_numeric(data, target, degree, gradient_method, descent_method, exact_method, lmda, eta, rs, ts, 
                  tol=1e-8, check=True,):
    """This is a generalized function of "theta_numeric" in part_e.

    Input: observed data, target, maximum polynomial degree, method to calculate gradient, 
    gradient descent method, method for exact theta, penalty parameter, learning rate, 
    random seed and test size. Default tolerence describes the convergence criteria for theta.
    
    Splits data into training and test data, calculates theta by gradient descent.
    """
    X = design_matrix(data, degree)
    X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

    if check:
        check_gradients(X_train_scaled, y_train_centered, lmda, rs, eta)

    # calculate exact theta
    theta_exact = exact_method(X_train_scaled, y_train_centered, lmda)

    # calculate numerical theta
    rng = np.random.default_rng(rs)
    theta0 = rng.normal(size=(degree))

    theta, iterations, history = gradient_descent_general(
        X_train_scaled, y_train_centered, theta0, gradient_method, 
        descent_method, lmda, eta, theta_exact, tol
        )
    return iterations, history


def plot_lambda_methods(x, y, degree, gradient_method, descent_methods, exact_method, lmdas, eta, rs, ts,
                        tol=1e-8, target_accuracy=1e-4, check=True, plot=True):
    """Plot iterations vs penalty parameter for several gradient descent methods.

    Top panel: iterations until the parameter update is smaller than tol.
    Bottom panel: iterations until theta is within target_accuracy of the closed-form solution.
    The learning rate eta is the initial/fixed step size used by the descent method.

    LLM-assisted
    ------------
    Tool: Cursor (October 2026)
    Role: The LLM wrote the entire function inspired by a previous draft written by myself.
    """
    fig, axs = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    ax1, ax2 = axs

    for descent_method, label, color in descent_methods:
        iterations = []
        iters_to_target = []
        for lmda in lmdas:
            iteration, history = theta_numeric(
                x, y, degree, gradient_method, descent_method, exact_method, lmda, eta, rs, ts, tol,
                check=check
            )
            iterations.append(iteration)

            diffs = np.array(history["theta_diff"])
            below = np.where(diffs < target_accuracy)[0]
            iters_to_target.append(below[0] + 1 if len(below) > 0 else np.nan)

        ax1.plot(lmdas, iterations, "o-", color=color, label=label)
        ax2.plot(lmdas, iters_to_target, "o-", color=color, label=label)

    ax1.set_ylabel(r'Iterations')
    ax1.set_yscale("log")
    ax1.set_title(rf"Convergence Speed ($\eta_0={eta}$)")
    ax1.legend()

    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(rf'Iterations')
    ax2.set_yscale("log")
    ax2.set_title(rf"Convergense to Closed-Form Solution ($\eta_0={eta}$)")
    ax2.legend(loc='lower left')

    plt.tight_layout()

    if plot:
        plt.show()
    else:
        return fig, axs


if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_automatic_diff
    descent_method = momentum
    exact_method = ols_ridge_exact
    ts = 0.2

    descent_methods = [
        (momentum, "Momentum", 'blue'),
        (AdaGrad, "AdaGrad", 'orange'),
        (RMSprop, "RMSprop", 'green'),
        (Adam, "Adam", 'red'),
    ]
    
    lmdas = np.arange(0.1, 1, 0.1)
    plot_lambda_methods(
        x, y, degree, gradient_method, descent_methods, exact_method, lmdas, eta=2, rs=rs, ts=ts
    )