import numpy as np
import matplotlib.pyplot as plt

def forward_euler(lam, r_func, y0, T, dt):
    """Forward Euler method"""
    t = np.arange(0, T + dt, dt)
    y = np.zeros_like(t)
    y[0] = y0
    
    for n in range(len(t) - 1):
        y[n+1] = (1 + lam * dt) * y[n] + dt * r_func(t[n])
    
    return t, y

def backward_euler(lam, r_func, y0, T, dt):
    """Backward Euler method (explicit solve for linear case)"""
    t = np.arange(0, T + dt, dt)
    y = np.zeros_like(t)
    y[0] = y0
    
    for n in range(len(t) - 1):
        y[n+1] = (y[n] + dt * r_func(t[n+1])) / (1 - lam * dt)
    
    return t, y

# Parameters
lam = -10
r_func = np.sin  # r(t) = sin(t)
y0 = 1
T = 5

# Test different time steps
dt_values = [0.1, 0.05, 0.01]

plt.figure(figsize=(12, 8))

for i, dt in enumerate(dt_values):
    # Forward Euler
    t_fe, y_fe = forward_euler(lam, r_func, y0, T, dt)
    
    # Backward Euler
    t_be, y_be = backward_euler(lam, r_func, y0, T, dt)
    
    # Plot
    plt.subplot(2, 2, i+1)
    plt.plot(t_fe, y_fe, 'b-', linewidth=2, label='Forward Euler')
    plt.plot(t_be, y_be, 'r--', linewidth=2, label='Backward Euler')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'Δt = {dt}')
    plt.grid(True)
    plt.legend()

# Compare both methods at smallest dt
dt = 0.01
t_fe, y_fe = forward_euler(lam, r_func, y0, T, dt)
t_be, y_be = backward_euler(lam, r_func, y0, T, dt)

plt.subplot(2, 2, 4)
plt.plot(t_fe, y_fe, 'b-', linewidth=2, label='Forward Euler')
plt.plot(t_be, y_be, 'r--', linewidth=2, label='Backward Euler')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'Comparison at smallest Δt = {dt}')
plt.grid(True)
plt.legend()

plt.suptitle('Forward vs Backward Euler for y\' = -10y + sin(t)', fontsize=14)
plt.tight_layout()
plt.savefig('euler_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute and display errors for the smallest dt
# Analytical solution (can be derived for this linear ODE)
from scipy.integrate import solve_ivp

def ode_system(t, y):
    return lam * y + np.sin(t)

sol = solve_ivp(ode_system, [0, T], [y0], t_eval=t_fe, method='RK45', rtol=1e-12)

print("\nNumerical solution at t = 5:")
print(f"Forward Euler:  y(5) = {y_fe[-1]:.6f}")
print(f"Backward Euler: y(5) = {y_be[-1]:.6f}")
print(f"Reference (RK45): y(5) = {sol.y[0,-1]:.6f}")

print(f"\nAbsolute errors at t = 5:")
print(f"Forward Euler error:  {abs(y_fe[-1] - sol.y[0,-1]):.3e}")
print(f"Backward Euler error: {abs(y_be[-1] - sol.y[0,-1]):.3e}")
