
from part_e import *


def momentum(theta, state, eta, grad, mom_par=0.9):
    change = eta*grad + mom_par*state.get('v', np.zeros_like(theta))
    theta = theta - change
    state = {'v': change}

    return theta, state


def AdaGrad(theta, state, eta, grad, eps=1e-8):
    r = state.get('r', np.zeros_like(theta)) + grad**2
    theta = theta - eta * grad / (np.sqrt(r) + eps) 
    state = {'r': r}

    return theta, state


def RMSprop(theta, state, eta, grad, rho=0.9, eps=1e-8):
    v = rho * state.get('v', np.zeros_like(theta)) + (1 - rho) * grad**2
    theta = theta - eta * grad / (np.sqrt(v) + eps)
    state = {'v': v}

    return theta, state


def Adam(theta, state, eta, grad, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state.get('t', 0) + 1
    m = (beta1 * state.get('m', np.zeros_like(theta)) + (1-beta1) * grad) 
    v = (beta2 * state.get('v', np.zeros_like(theta)) + (1-beta2) * grad**2) 
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)
    state = {'m': m, 'v': v, 't': t}

    return theta, state


def gradient_descent_general(X, y, theta, grad_meth, descent_method, lmda, eta, theta_exact, tol=1.0e-8):
    """Returns theta calculated by gradient descent, 
    and the number of itterations. 

    The gradient methods available is: OLS, Ridge, automatic differentiation
    The descent methods available is: momentum, AdaGrad
    """
    state = {}
    for k in range(100000):
        gradient = grad_meth(theta, X, y, lmda)
        theta, state = descent_method(theta, state, eta, gradient)

        if np.linalg.norm(gradient) < tol:
            break

    difference = float(np.linalg.norm(theta - theta_exact))

    return theta, k+1, difference


def theta_numeric(data, target, degree, gradient_method, descent_method, lmda, eta, rs, ts):
    """Input: observed data, target, maximum polynomial degree, 
    method of which to calculate gradient, penalty scalar, learning rate, 
    random seed and test size
    
    Splits into training and test data, calculates theta by gradient descent.
    Prints difference between numerical and exact theta.
    """
    X = design_matrix(data, degree)
    X_train_scaled, X_test_scaled, y_train_centered, y_test_centered = split_scale(X, target, ts, rs)

    # check of gradients and learning rate
    check_gradients(X_train_scaled, y_train_centered, lmda, rs, eta)

    # calculate exact theta
    n = len(y_train_centered)
    theta_exact = np.linalg.pinv(X_train_scaled.T @ X_train_scaled + n*lmda*np.eye(X_train_scaled.shape[1])) @ X_train_scaled.T @ y_train_centered

    # calculate numerical theta
    rng = np.random.default_rng(rs)
    theta0 = rng.normal(size=(degree))
    theta, iterations, difference = gradient_descent_general(X_train_scaled, y_train_centered, theta0, gradient_method, descent_method, lmda, eta, theta_exact)

    # compares analytical and numerical theta
    print(f"Difference between exact and numerical theta:   {theta_exact - theta}")

    return iterations


def plot_eta_lambda(x, y, degree, gradient_method, descent_method, etas, lmdas, rs, ts):
    """Written with help of AI"""
    fig, ax = plt.subplots()
    for lmda in lmdas:
        iterations = []
        for eta in etas:
            iteration = theta_numeric(x, y, degree, gradient_method, descent_method, lmda, eta, rs, ts)
            iterations.append(iteration)
        ax.plot(etas, iterations, 'o-', label=rf'$\lambda={lmda}$')

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'iterations to converge to $10^{-8}$')
    ax.set_yscale('log')
    ax.set_title('Effect of learning rate and lambda on convergence speed')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_automatic_diff  
    descent_method = RMSprop
    ts = 0.2

    etas = np.linspace(0.02, 0.35, 8)
    lmdas = [0.0, 0.01, 0.1, 1.0]

    plot_eta_lambda(x, y, degree, gradient_method, descent_method, etas, lmdas, rs, ts)