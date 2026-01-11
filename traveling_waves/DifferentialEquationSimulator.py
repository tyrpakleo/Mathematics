import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt

__all__ = ['DifferentialEquationSimulator']


class DifferentialEquationSimulator:
    """
    Simulate 2D differential equations in polar or Cartesian coordinates.

    Handles ODE integration and phase-plane visualization with optional
    timestamp markers. Supports both coordinate systems with shared plotting logic.
    """

    def __init__(self, num_points=1000, timestamps=0):
        """
        Initialize the simulator with default parameters.

        Parameters
        ----------
        num_points : int, optional
            Number of time points to evaluate (default: 1000).
        timestamps : int, optional
            Number of timestamp crosses to display (default: 0).
        """
        self.num_points = num_points
        self.timestamps = timestamps

    def _timestamp_indices(self, n, length):
        """Compute indices for timestamp markers evenly across time."""
        if n <= 0:
            return []
        idxs = np.linspace(0, length - 1, n + 2)[1:-1]
        return np.round(idxs).astype(int)

    def _plot_cartesian(self, ax, x, y, t):
        """Plot Cartesian phase plane with optional timestamp crosses."""
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
        if self.timestamps > 0:
            for i, idx in enumerate(self._timestamp_indices(self.timestamps, len(t)), start=1):
                ax.plot(x[idx], y[idx], 'kx', markersize=12, markeredgewidth=2)
                ax.text(x[idx], y[idx], f' {i}', fontsize=10, verticalalignment='bottom')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Phase Space (Cartesian)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

    def _plot_polar(self, ax, theta, r, t):
        """Plot polar phase plane with optional timestamp crosses."""
        ax.plot(theta, r, 'b-', linewidth=1.5, alpha=0.7)
        ax.plot(theta[0], r[0], 'go', markersize=10, label='Start')
        ax.plot(theta[-1], r[-1], 'ro', markersize=10, label='End')
        if self.timestamps > 0:
            for i, idx in enumerate(self._timestamp_indices(self.timestamps, len(t)), start=1):
                ax.plot(theta[idx], r[idx], 'kx', markersize=12, markeredgewidth=2)
                ax.text(theta[idx], r[idx], f' {i}', fontsize=10, verticalalignment='bottom')
        ax.set_xlabel('θ (radians)')
        ax.set_ylabel('r')
        ax.set_title('Phase Space (Polar)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def simulate_polar(self, r0, theta0, t_max, dr_dt_func, dtheta_dt_func):
        """
        Simulate a 2D ODE given in polar coordinates.

        Parameters
        ----------
        r0 : float
            Initial radial coordinate.
        theta0 : float
            Initial angular coordinate (radians).
        t_max : float
            Maximum time to simulate.
        dr_dt_func : callable
            Function (r, theta, t) -> dr/dt.
        dtheta_dt_func : callable
            Function (r, theta, t) -> dtheta/dt.

        Returns
        -------
        t, r, theta, x, y : ndarray
            Time, polar (r, theta), and Cartesian (x, y) arrays.
        """
        t = np.linspace(0, t_max, self.num_points)

        def polar_system(state, tt):
            r, theta = state
            return [dr_dt_func(r, theta, tt), dtheta_dt_func(r, theta, tt)]

        solution = odeint(polar_system, [r0, theta0], t)
        r = solution[:, 0]
        theta = solution[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self._plot_cartesian(ax1, x, y, t)
        self._plot_polar(ax2, theta, r, t)
        plt.tight_layout()
        plt.show()

        return t, r, theta, x, y

    def simulate_cartesian(self, x0, y0, t_max, dx_dt_func, dy_dt_func):
        """
        Simulate a 2D ODE given in Cartesian coordinates.

        Parameters
        ----------
        x0 : float
            Initial x coordinate.
        y0 : float
            Initial y coordinate.
        t_max : float
            Maximum time to simulate.
        dx_dt_func : callable
            Function (x, y, t) -> dx/dt.
        dy_dt_func : callable
            Function (x, y, t) -> dy/dt.

        Returns
        -------
        t, r, theta, x, y : ndarray
            Time, derived polar (r, theta), and Cartesian (x, y) arrays.
        """
        t = np.linspace(0, t_max, self.num_points)

        def cartesian_system(state, tt):
            x, y = state
            return [dx_dt_func(x, y, tt), dy_dt_func(x, y, tt)]

        solution = odeint(cartesian_system, [x0, y0], t)
        x = solution[:, 0]
        y = solution[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        self._plot_cartesian(ax, x, y, t)
        plt.tight_layout()
        plt.show()

        return t, r, theta, x, y


if __name__ == "__main__":
    # Example: Spiral system
    # dr/dt = r*(4 - r^2)
    # dtheta/dt = 1.0

    def dr_dt(r, theta, t):
        return r * (4 - r**2)

    def dtheta_dt(r, theta, t):
        return 1.0

    def dx_dt(x, y, t):
        return y

    def dy_dt(x, y, t):
        a = 1.0
        return -x - a * y * (x**2 + y**2 - 1)

    # Initial conditions
    r0 = 0.1
    theta0 = 0.0
    t_max = 20.0
    time_stamps = 5

    # Create simulator instance
    sim = DifferentialEquationSimulator(num_points=1000, timestamps=time_stamps)

    # # Run simulation (polar)
    # t, r, theta, x, y = sim.simulate_polar(
    #     r0=r0,
    #     theta0=theta0,
    #     t_max=t_max,
    #     dr_dt_func=dr_dt,
    #     dtheta_dt_func=dtheta_dt,
    # )

    # print("Simulation complete!")
    # print(f"Initial: r={r0}, θ={theta0}")
    # print(f"Final: r={r[-1]:.4f}, θ={theta[-1]:.4f}")

    x0 = 0.01
    y0 = 0.01

    # Run simulation (Cartesian)
    t, r, theta, x, y = sim.simulate_cartesian(
        x0=x0,
        y0=y0,
        t_max=t_max,
        dx_dt_func=dx_dt,
        dy_dt_func=dy_dt,
    )

    print("Simulation complete!")
    print(f"Initial: x={x0}, y={y0}")
    print(f"Final: x={x[-1]:.4f}, y={y[-1]:.4f}")