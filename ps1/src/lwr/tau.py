import os

import matplotlib.pyplot as plt
import numpy as np
import util
from lwr import LocallyWeightedLinearRegression
from pathlib import Path # Just for DATA_DIR

def createVisual(mse, x, y, preds):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x[:, 1], y, "bx", markersize=5, markeredgewidth=0.5, label="Validation")
    ax.plot(x[:, 1], preds, "ro", markersize=3, markeredgewidth=0.2, label="Predictions")
    ax.set_title("Validation & Predictions", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for handle in legend.legendHandles:
        handle.set_markersize(7)
        handle.set_markeredgewidth(1.0)

    fig.text(0.5, 0.035, f"MSE: {mse:.10f}", ha="center", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.28)
    return fig

def calculations(x, x_train, y_train, tau, clf):
    # Norm accross the feature dimension (in this case it is dimension 1, so it just does absolute value)
    distances = np.linalg.norm(x[:, None] - x_train[None, :], axis=-1)

    # Each row is the diagonal of the W matrix of a) for each evaluation point
    weights = np.exp(-distances**2/(2*tau**2)) 

    predictions = np.empty(x.shape[0])

    for i, ev_point in enumerate(x):
        weight_vector = weights[i]
        Wx = weight_vector[:, None] * x_train
        Wy = weight_vector * y_train

        A = x_train.T @ Wx
        b = x_train.T @ Wy

        # Instead of computing inverse, we solve a
        # system of equations
        clf.last_theta = np.linalg.solve(A, b)
        predictions[i] = clf.predict(ev_point)
    return predictions

def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
        
    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    lowest_mse = float("inf")
    arg_tau_l_mse = -1
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)

        preds = calculations(x_eval, x_train, y_train, tau, clf)
        mse = calculate_mse(y_eval, preds)

        if lowest_mse > mse:
            lowest_mse = mse
            arg_tau_l_mse = tau

        save_plot(createVisual(mse, x_eval, y_eval, preds), tau)

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    preds = calculations(x_test, x_train, y_train, arg_tau_l_mse, LocallyWeightedLinearRegression(arg_tau_l_mse))
    print(f"Min. MSE on valid split (tau = {arg_tau_l_mse}): {lowest_mse} \n"
          f"MSE on test split with tau = {arg_tau_l_mse}: {calculate_mse(y_test, preds)}")
    
# *** END CODE HERE ***

def save_plot(fig, tau):
    plots_dir = DATA_DIR / "tau_plots"
    plots_dir.mkdir(exist_ok=True)
    fig_path = plots_dir / f"plot-tau-{tau}.png"
    fig.savefig(fig_path)

def calculate_mse(y, pred):
    return np.mean((pred - y)**2)

DATA_DIR = Path(__file__).parent

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path= DATA_DIR/'train.csv',
         valid_path= DATA_DIR/'valid.csv',
         test_path= DATA_DIR/'test.csv',
         pred_path= DATA_DIR/'pred.txt')
