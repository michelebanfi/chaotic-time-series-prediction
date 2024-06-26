from scipy.integrate import solve_ivp
import numpy as np

# general function to generate data
def generate_data(equation, t_span, y0, t_eval):
    sol = solve_ivp(equation, t_span, y0, t_eval=t_eval)
    return sol.t, sol.y.T

# Lorenz system
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Harmonic oscillator
def harmonic_oscillator(t, state, k=1, m=1):
    x, v = state
    dxdt = v
    dvdt = -k / m * x
    return [dxdt, dvdt]

# Restricted three-body problem
def restricted_three_body(t, state):
    # x, y, vx, vy
    x, y, vx, vy = state
    m1 = 1
    m2 = 1
    x1 = 1
    y1 = 1
    x2 = -1
    y2 = -1
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    ax = - m1 * ((x - x1) / r1 ** 3) - m2 * ((x - x2) / r2 ** 3)
    ay = - m1 * ((y - y1) / r1 ** 3) - m2 * ((y - y2) / r2 ** 3)
    return [vx, vy, ax, ay]