"""
The main script for calibration.
"""

import time
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize
# import seaborn as sns

from pricing import heston_price_vanilla_row

def calibrate_model_row(row):
    """
    Calibrate the Heston model using market data.

    This method performs the following steps:
    1. Retrieves the spot price and interpolates necessary market data.
    2. Sets up the optimization problem with initial parameter guesses and bounds.
    3. Defines the objective function to minimize the squared difference between
    model prices and market prices.
    4. Uses the SLSQP algorithm to find the optimal model parameters.

    Returns:
        tuple: A tuple containing two elements:
            - result (OptimizeResult): The optimization result from scipy.optimize.minimize.
            - parameter_history (list): A list of parameter sets explored during optimization.

    Raises:
        Any exceptions raised by the optimisation process are not explicitly handled.
    """
    print("Calibrating model")

    params = {
        "v0": {"x0": 0.1, "bounds": (1e-2, 1)},
        "kappa": {"x0": 3, "bounds": (1e-2, 5)},
        "theta": {"x0": 0.05, "bounds": (1e-2, 1)},
        "sigma": {"x0": 0.3, "bounds": (1e-2, 2)},
        "rho": {"x0": -0.8, "bounds": (-1, 1)},
        "lambda_": {"x0": 0.03, "bounds": (-1, 1)},
    }
    initial_values = np.array([param["x0"] for param in params.values()])
    bounds = [param["bounds"] for param in params.values()]

    def callback(x):
        """
        Callback function to record parameter history during optimization.

        Args:
            x (np.array): Current parameter set being evaluated.
        """
        self.parameter_history.append(x.copy())

    def fellers_constraint(x):
        """
        Feller's contstraint ensures non-negative variance.
        """
        _, kappa, theta, sigma, _, _ = x
        return 2 * kappa * theta - sigma**2

    def objective_function(x):
        """
        Objective function to be minimized in the calibration process.

        This function calculates the sum of squared differences between
        model prices and market prices, with a small regularization term.

        Args:
            x (np.array): Array of model parameters [v0, kappa, theta, sigma, rho, lambda_].

        Returns:
            float: The value of the objective function to be minimized.
        """
        v0 = row['v0']
        kappa = row['kappa']
        theta = row['theta']
        sigma = row['sigma']
        rho  = row['rho']
        v0, kappa, theta, sigma, rho, lamba_ = x
        model_prices['heston_price'] = heston_price_vanilla_row(row)

        mse = np.sum((model_prices - option_prices)**2) / (len(strikes) * len(maturities))

        # Adding a penalty for violating Feller's condition
        feller_violation = max(0, sigma**2 - 2 * kappa * theta)
        penalty = 1e6 * feller_violation  # Large penalty for violation
        return mse + penalty + 1e-4 * np.sum(x**2)

    fellers_constraint_dict = {"type": "ineq", "fun": fellers_constraint}

    result = minimize(objective_function, initial_values, method='SLSQP',
                bounds=bounds, constraints=[fellers_constraint_dict],
                options={'ftol': 1e-4, 'maxiter': 1000}, callback=callback)

    if result.success:
        print("Optimisation successful.")
    else:
        print("Optimisation failed. Reason:", result.message)

    return result



# =============================================================================
#     def plot_calibration_result(self):
#         """
#         Plots the convergence of the optimisation function.
#         """
#     # Check if the model has been calibrated
#         if not hasattr(self, 'parameter_history') or self.parameter_history is None:
#             raise RuntimeError("Model calibration has not been performed yet. Please run 'calibrate_model' first.")
#         
#         param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambda_']
#         param_history = np.array(self.parameter_history)
# 
#         # Set the style and color palette
#         sns.set(style="whitegrid", palette="deep")
# 
#         fig, axs = plt.subplots(3, 2, figsize=(15, 10))
#         fig.suptitle('Heston Model Parameter Convergence', fontsize=20, y=1.02)
# 
#         for i, (ax, name) in enumerate(zip(axs.flatten(), param_names)):
#             sns.lineplot(data=param_history[:, i], ax=ax, linewidth=2, marker='o')
#             ax.set_title(f'{name.capitalize()} Convergence', fontsize=14)
#             ax.set_xlabel('Iteration', fontsize=12)
#             ax.set_ylabel('Parameter Value', fontsize=12)
#             ax.tick_params(labelsize=10)
# 
#             # Add final value annotation
#             final_value = param_history[-1, i]
#             ax.annotate(f'Final: {final_value:.4f}',
#                         xy=(len(param_history)-1, final_value),
#                         xytext=(0.7, 0.95),
#                         textcoords='axes fraction',
#                         fontsize=10,
#                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
#                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
# 
#         plt.tight_layout()
#         fig.subplots_adjust(top=0.93)  # Adjust to prevent title overlap
# 
#         # Add a text box with calibration summary
#         convergence_text = (f"Calibration converged in {len(self.parameter_history)} iterations.\n"
#                             f"Final parameter values:\n"
#                             f"v0: {param_history[-1, 0]:.4f}\n"
#                             f"kappa: {param_history[-1, 1]:.4f}\n"
#                             f"theta: {param_history[-1, 2]:.4f}\n"
#                             f"sigma: {param_history[-1, 3]:.4f}\n"
#                             f"rho: {param_history[-1, 4]:.4f}\n"
#                             f"lambda: {param_history[-1, 5]:.4f}")
# 
#         fig.text(0.5, -0.05, convergence_text, ha='center', va='center', fontsize=12,
#                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
# 
#         plt.savefig('heston_parameter_convergence.png', dpi=300, bbox_inches='tight')
#         plt.show()
# =============================================================================
