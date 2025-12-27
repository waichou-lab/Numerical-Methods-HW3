import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    """Rosenbrock function"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    """Gradient of Rosenbrock function"""
    z = x[1] - x[0]**2
    gx = -400 * x[0] * z - 2 * (1 - x[0])
    gy = 200 * z
    return np.array([gx, gy])

def bfgs_rosenbrock(x0, max_iter=5000, tol=1e-8, verbose=True):
    """BFGS with backtracking Armijo line search"""
    # Parameters for Armijo
    c1 = 1e-4
    beta = 0.5
    
    x = x0.copy()
    H = np.eye(2)  # Initial inverse Hessian approximation
    
    # Storage for history
    hist_f = []
    hist_grad = []
    hist_alpha = []
    
    for k in range(max_iter):
        fk = rosenbrock(x)
        gk = rosenbrock_grad(x)
        grad_norm = np.linalg.norm(gk)
        
        hist_f.append(fk)
        hist_grad.append(grad_norm)
        
        # Check convergence
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iteration {k}")
            break
        
        # BFGS direction
        pk = -H @ gk
        
        # Backtracking Armijo line search
        alpha = 1.0
        gTp = gk @ pk
        
        # Ensure sufficient decrease
        while rosenbrock(x + alpha * pk) > fk + c1 * alpha * gTp:
            alpha *= beta
            if alpha < 1e-14:
                print("Warning: step size too small")
                break
        
        hist_alpha.append(alpha)
        
        # Update x
        x_new = x + alpha * pk
        g_new = rosenbrock_grad(x_new)
        
        # Update inverse Hessian approximation
        s = x_new - x
        y = g_new - gk
        ys = y @ s
        
        if ys > 1e-14:  # Safeguard
            rho = 1.0 / ys
            I = np.eye(2)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            H = np.eye(2)  # Reset if curvature condition fails
        
        x = x_new
    
    return x, hist_f, hist_grad, hist_alpha

# Run BFGS on Rosenbrock
x0 = np.array([-1.2, 1.0])
x_opt, f_hist, grad_hist, alpha_hist = bfgs_rosenbrock(x0)

print(f"Optimal point: x = {x_opt[0]:.8f}, y = {x_opt[1]:.8f}")
print(f"Function value: {rosenbrock(x_opt):.3e}")
print(f"Gradient norm: {np.linalg.norm(rosenbrock_grad(x_opt)):.3e}")
print(f"Iterations: {len(f_hist)}")

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot f(x_k)
axes[0].semilogy(range(len(f_hist)), f_hist, 'o-', markersize=4)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('$f(x_k)$')
axes[0].grid(True)
axes[0].set_title('Function value convergence')

# Plot gradient norm
axes[1].semilogy(range(len(grad_hist)), grad_hist, 'o-', markersize=4)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('$||\\nabla f(x_k)||$')
axes[1].grid(True)
axes[1].set_title('Gradient norm convergence')

plt.tight_layout()
plt.savefig('bfgs_convergence.png', dpi=150, bbox_inches='tight')
plt.show()