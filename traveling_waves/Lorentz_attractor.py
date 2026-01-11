import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz vector field
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def simulate(initial, t_eval, params):
    sigma, rho, beta = params
    sol = solve_ivp(lambda t, y: lorenz(t, y, sigma, rho, beta),
                    (t_eval[0], t_eval[-1]), initial, t_eval=t_eval, method='DOP853',
                    rtol=1e-9, atol=1e-12)
    return sol

params = (10.0, 28.0, 8.0/3.0)
t_eval = np.linspace(0, 40, 5000)

initials = [
    [1.0, 1.0, 1.0],
    [1.000001, 1.0, 1.0],  # tiny perturbation
    [0.9, 1.0, 1.0]
]

fig, ax = plt.subplots(figsize=(8,5))
for idx, y0 in enumerate(initials):
    sol = simulate(y0, t_eval, params)
    ax.plot(sol.t, sol.y[0], label=f'init {idx+1}')   # plot x(t) for each
ax.set_xlabel('time'); ax.set_ylabel('x(t)')
ax.legend()
ax.set_title('Lorenz: x(t) for different initial conditions')
plt.show()
