from ruptures.costs import BaseCost
import numpy as np
from math import log
import matplotlib.pyplot as plt
import ruptures as rpt


class CostPoisson(BaseCost):
    """Cost function for Poisson-distributed data.

    The cost is defined as the negative log-likelihood of the Poisson distribution
    for a given segment of data.

    Cost function is taken from Truong et. al 2019, doi: 10.1016/j.sigpro.2019.107299.
    """

    model = ""
    min_size = 1

    def fit(self, signal):
        """Fit the cost function to the signal.

        Parameters
        ----------
        signal : 1D array-like, shape (n_samples,)
            The input signal to fit the cost function to.
        """
        self.signal = np.asarray(signal)
        return self

    def error(self, start, end):
        """Compute the cost for a segment of the signal.

        Parameters
        ----------
        start : int
            The starting index of the segment (inclusive).
        end : int
            The ending index of the segment (exclusive).

        Returns
        -------
        float
            The computed cost for the segment.
        """
        sub = self.signal[start:end]

        # Calculate the mean of the segment
        mean_segment = np.mean(sub, axis=0)

        # Avoid log(0) by replacing zeros with a small value
        mean_segment = np.where(mean_segment == 0, 1e-10, mean_segment)

        # Compute the negative log-likelihood for Poisson distribution (simplified to remove additive constants)
        cost = -(end - start) * (mean_segment * log(mean_segment))
        return cost


def detect_poisson_change_points(count_values, time_points, penalty):
    """
    Use the Pelt algorithm to detect changepoints in counts data.
    Parameters
    ----------
    count_values : list or np.array
        The counts data to analyze.
    time_points : list or np.array
        The corresponding time points for the counts data.
    penalty : float
        The penalty value for the Pelt algorithm.
    Returns
    -------
    list
        The indices of detected change points, including the end of the signal.

    """
    algo = rpt.Pelt(custom_cost=CostPoisson()).fit(count_values)
    penalty = 1
    result = algo.predict(pen=penalty)
    print("Detected change points at:", [time_points[i] for i in result[:-1]])
    return result


def plot_change_points(time_points, count_values, change_points):
    """
    Plot the counts data with detected change points.
    Parameters
    ----------
    time_points : list or np.array
        The corresponding time points for the counts data.
    count_values : list or np.array
        The counts data to analyze.
    change_points : list
        The indices of detected change points.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, count_values, label="Count Data")
    for cp in change_points[:-1]:
        plt.axvline(
            x=time_points[cp],
            color="red",
            linestyle="--",
            label="Change Point" if cp == change_points[0] else "",
        )
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
