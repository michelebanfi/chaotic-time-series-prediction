from scipy.integrate import solve_ivp
import numpy as np

# general function to generate data
def generate_data(equation, t_span, y0, t_eval, args=()):
    sol = solve_ivp(equation, t_span, y0, t_eval=t_eval, args=args)
    print(sol.message)
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
def restricted_three_body(t, state, m1, m2, x1, y1, x2, y2):
    # x, y, vx, vy
    x, y, vx, vy = state
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + 1e-30
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2) + 1e-30
    ax = - m1 * ((x - x1) / r1 ** 3) - m2 * ((x - x2) / r2 ** 3)
    ay = - m1 * ((y - y1) / r1 ** 3) - m2 * ((y - y2) / r2 ** 3)
    return [vx, vy, ax, ay]