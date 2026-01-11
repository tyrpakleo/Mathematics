import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ODE definition
def decay(t, y):
    k = 0.5
    return (y-k) * y

# Time span and initial condition
t_span = (0, 10)
y0_1 = [0.499]

# Solve ODE
solution1 = solve_ivp(
    decay,
    t_span,
    y0_1,
    t_eval=np.linspace(0, 10, 200)
)

# Plot result
plt.plot(solution1.t, solution1.y[0])

plt.xlabel("Time")
plt.ylabel("y(t)")
plt.show()