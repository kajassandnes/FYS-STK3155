
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


def theta_numeric(data, target, degree, gradient_method, descent_method, exact_method, lmda, eta, rs, ts, tol=1e-8):
    """This is a generalized function of "theta_numeric" in part_e.

    Input: observed data, target, maximum polynomial degree, method to calculate gradient, 
    gradient descent method, method for exact theta, penalty parameter, learning rate, 
    random seed and test size. Default tolerence describes the convergence criteria for theta.
    
    Splits data into training and test data, calculates theta by gradient descent.
    Prints difference between numerical and exact theta.
    """
    X = design_matrix(data, degree)
    X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

    # check of gradient
    check_gradients(X_train_scaled, y_train_centered, lmda, rs, eta)

    # calculate exact theta
    theta_exact = exact_method(X_train_scaled, y_train_centered, lmda)

    # calculate numerical theta
    rng = np.random.default_rng(rs)
    theta0 = rng.normal(size=(degree))
    theta, iterations, history = gradient_descent_general(X_train_scaled, y_train_centered, theta0, gradient_method, descent_method, lmda, eta, theta_exact, tol)

    # compares analytical and numerical theta
    print(f"eta: {eta}  lmda: {lmda}")
    print(f"analytical theta:                               {theta_exact.ravel()}")
    print(f"Difference between exact and numerical theta:   {theta_exact - theta}")
    print(f"Difference: {np.linalg.norm(theta_exact) - np.linalg.norm(theta)}")

    return iterations, history


def plot_eta_lambda(x, y, degree, gradient_method, descent_method, exact_method, lmdas, etas, rs, ts, tol=1e-8, target_accuracy=1e-4):
    """This is a generalized function of "plot_eta_lambda" in part_e.
    
    Inputs: dataset, target, maximum polynomial degree, method to compute gradient, 
    gradient descent method, method for exact theta, penalty parameters, learning rates
    random seed, test size. Default tolerance determines the criteria of convergence of 
    theta during learning. Default target accuracy describes the criteria of convergense 
    to closed form solution.

    The function plots iterations to convergence and closed form solution vs learning rate
    for different values of the penalty parameter.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    for lmda in lmdas:
        iterations = []
        iters_to_target = []
        for eta in etas:
            iteration, history = theta_numeric(x, y, degree, gradient_method, descent_method, exact_method, lmda, eta, rs, ts, tol)
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
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel(r'iterations to converge (tol = $10^{-8}$)')
    ax1.set_yscale('log')
    ax1.set_title('Effect of learning rate and lambda on convergence speed')
    ax1.legend()

    # iterations to converge to closed solution
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel(rf'iterations to reach $\|\theta_{{exact}} - \theta\| < {target_accuracy:.0e}$')
    ax2.set_yscale('log')
    ax2.set_title('Iterations to reach closed-form solution')
    ax2.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_automatic_diff  
    descent_method = RMSprop
    exact_method = ols_ridge_exact
    ts = 0.2

    etas = np.linspace(0.001, 3.0, 20)
    lmdas = [0, 0.01, 0.1, 1.0]

    plot_eta_lambda(x, y, degree, gradient_method, descent_method, exact_method, lmdas, etas, rs, ts)
    # kan plotte de ulike metodene for en fast lmda mot hverandre