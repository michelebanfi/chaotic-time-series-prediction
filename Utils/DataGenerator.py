from scipy.integrate import solve_ivp

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
def restricted_three_body(t, state, mu=0.012277471):
    x, y, vx, vy = state
    r1 = ((x + mu) ** 2 + y ** 2) ** 1.5
    r2 = ((x - 1 + mu) ** 2 + y ** 2) ** 1.5
    dvxdt = 2 * vy + x - (1 - mu) * (x + mu) / r1 - mu * (x - 1 + mu) / r2
    dvydt = -2 * vx + y - (1 - mu) * y / r1 - mu * y / r2
    return [vx, vy, dvxdt, dvydt]