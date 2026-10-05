from part_f import *


def step_length(t, t0, t1):
    """Learning-rate schedule from the lecture notes: eta = t0 / (t + t1)."""
    return t0 / (t + t1)


def stochastic_gradient_descent(X, y, theta, descent_method, gradient_method, lmda, M, n_epochs, t0, t1):
    """Split the training data into minibatches and loop over epochs.

    Follows the lecture-notes structure: m = n/M minibatches, each of size M.
    In every epoch a minibatch B_k is drawn at random and theta is updated with
    the chosen descent method. Returns the final theta and theta after every epoch.
    """
    n = X.shape[0]
    m = int(n / M)

    B = np.split(X[:m * M], m)
    y_b = np.split(np.ravel(y)[:m * M], m)
    state = {}
    thetas = []

    for epoch in range(1, n_epochs + 1):
        for i in range(m):
            k = np.random.randint(m)
            t = epoch * m + i
            eta = step_length(t, t0, t1)
            grad = gradient_method(theta, B[k], y_b[k], lmda*M/n)
            theta, state = descent_method(theta, state, eta, grad)
        thetas.append(theta)

    return theta, thetas


def plot_dependence(X, y, theta0, theta_exact, methods, lmda, Ms, epochs, M_fixed, epochs_fixed, t1, rs):
    """Left: error vs mini-batch size. Right: error vs number of epochs.
    
    LLM-assisted
    ------------
    Tool: Cursor (October 2026)
    Role: The LLM wrote the entire function after promt from myself.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for descent_method, label, color, eta0 in methods:
        t0 = eta0 * t1          # first step is roughly eta0

        errors = []
        for M in Ms:
            np.random.seed(rs)
            theta, _ = stochastic_gradient_descent(
                X, y, theta0.copy(), descent_method, gradient_automatic_diff, lmda, M, epochs_fixed, t0, t1)
            errors.append(np.linalg.norm(theta_exact - theta))
        ax1.plot(Ms, errors, "o-", color=color, label=label)

        np.random.seed(rs)
        _, thetas = stochastic_gradient_descent(
            X, y, theta0.copy(), descent_method, gradient_automatic_diff, lmda, M_fixed, max(epochs), t0, t1)
        errors = [np.linalg.norm(theta_exact - thetas[e - 1]) for e in epochs]
        ax2.plot(epochs, errors, "o-", color=color, label=label)

    ax1.set_xlabel("Mini-batch size $M$")
    ax1.set_ylabel(r"$\|\theta_{exact} - \theta\|$")
    ax1.set_yscale("log")
    ax1.set_title(f"{epochs_fixed} epochs")
    ax1.legend()

    ax2.set_xlabel("Epochs")
    ax2.set_title(f"$M={M_fixed}$")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rs = 2026
    x, y = runge_data(rs=rs)
    degree = 5
    ts = 0.2
    lmda = 0.1

    X = design_matrix(x, degree)
    X_train, X_test, y_train, y_test = split_scale(X, y, ts, rs)

    theta_exact = ols_ridge_exact(X_train, y_train, lmda)
    theta0 = np.random.default_rng(rs).normal(size=X_train.shape[1])

    methods = [
        (momentum, "Momentum", "blue", 0.01),
        (AdaGrad, "AdaGrad", "orange", 2.0),
        (RMSprop, "RMSprop", "green", 0.03),
        (Adam, "Adam", "red", 0.05),
    ]

    Ms = np.arange(1, 51, 1)
    epochs = np.arange(10, 2600, 100)

    plot_dependence(X_train, y_train, theta0, theta_exact, methods, lmda, Ms, epochs,
                    M_fixed=5, epochs_fixed=500, t1=1000, rs=rs)
