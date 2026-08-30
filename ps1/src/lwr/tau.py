import os

import matplotlib.pyplot as plt
import numpy as np
import util
from lwr import LocallyWeightedLinearRegression


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
    
# TODO: implement this section.
pass
# *** END CODE HERE ***

def save_plot(fig, tau):
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    fig_path = os.path.join(plots_dir, f"plot-tau-{tau}.png")
    fig.savefig(fig_path)

def calculate_mse(y, pred):
    return np.mean((pred - y)**2)
    
if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
