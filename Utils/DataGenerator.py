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
