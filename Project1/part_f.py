
from part_e import *

def momentum(X, y, theta, grad_meth, lmda, eta, theta_exact, tol=1.0e-8, extras=None):
    """Returns theta calculated by gradient descent, 
    and the number of itterations. 

    The gradient methods available is: OLS, Ridge, automatic differentiation.
    """
    mom = extras[0]
    change = 0.0
    for k in range(100000):
        gradient = grad_meth(theta, X, y, lmda)
        new_change = eta * gradient + mom * change
        theta = theta - new_change
        change = new_change

        if np.linalg.norm(gradient) < tol:
            break

    difference = float(np.linalg.norm(theta - theta_exact))

    return theta, k+1, difference



if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    gradient_method = gradient_automatic_diff  
    descent_method = momentum
    ts = 0.2

    etas = np.linspace(0.02, 0.35, 8)
    lmdas = [0.0, 0.01, 0.1, 1.0]

    plot_eta_lambda(x, y, degree, gradient_method, descent_method, etas, lmdas, rs, ts, [0.1])