import matplotlib.pyplot as plt
import numpy as np
import util
from pathlib import Path # Just for DATA_DIR

def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    # Norm accross the feature dimension (in this case it is dimension 1, so it just does absolute value)
    distances = np.linalg.norm(x_eval[:, None] - x_train[None, :], axis=-1)

    # Each row is the diagonal of the W matrix of a) for each evaluation point
    weights = np.exp(-distances**2/(2*tau**2)) 

    predictions = np.empty(x_eval.shape[0])

    for i, ev_point in enumerate(x_eval):
        weight_vector = weights[i]
        Wx = weight_vector[:, None] * x_train
        Wy = weight_vector * y_train

        A = x_train.T @ Wx
        b = x_train.T @ Wy

        # Instead of computing inverse, we solve a
        # system of equations
        clf.last_theta = np.linalg.solve(A, b)
        predictions[i] = clf.predict(ev_point)

    mse = np.average((y_eval - predictions)**2)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_eval[:, 1], y_eval, "bx", markersize=5, markeredgewidth=0.5, label="Validation")
    ax.plot(x_eval[:, 1], predictions, "ro", markersize=3, markeredgewidth=0.2, label="Predictions")
    ax.set_title("Validation & Predictions", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for handle in legend.legendHandles:
        handle.set_markersize(7)
        handle.set_markeredgewidth(1.0)

    fig.text(0.5, 0.035, f"MSE: {mse:.10f}", ha="center", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(DATA_DIR / "lwr-validation.png", bbox_inches="tight")
    return mse
# *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None
        self.last_theta = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
# *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.last_theta @ x

# *** END CODE HERE ***

DATA_DIR = Path(__file__).parent

if __name__ == '__main__':
    main(tau=5e-1,
         train_path=DATA_DIR / 'train.csv',
         eval_path=DATA_DIR / 'valid.csv')
