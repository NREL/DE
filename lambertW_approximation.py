import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# Define the custom Lambert W function using Newton-Raphson as specified by the user
def lambertw_newton(z, tol=1e-6, max_iter=10):
    w = np.log(z + 1)  # Initial guess based on approximation
    for i in range(max_iter):
        ew = np.exp(w)
        w_next = w - (w * ew - z) / (ew + w * ew)
        if np.all(np.abs(w_next - w) < tol):
            return w_next
        w = w_next
    return w  # Return the last approximation if convergence criteria are not met

# Test the function on a range of values and compare with scipy's implementation
z_values = np.logspace(-25, 6, 100)  # Avoid starting exactly at 0 to prevent log(1) -> 0 initial guess
w_custom = np.array([lambertw_newton(z) for z in z_values])
w_scipy = lambertw(z_values).real

# Calculate the absolute errors between the implementations
errors = np.abs(w_custom - w_scipy)

# Plot the results and the error
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(z_values, w_custom, label='Custom Newton-Raphson', color='red')
plt.plot(z_values, w_scipy, label='SciPy LambertW', linestyle='--', color='blue')
plt.title('Comparison of Lambert W Implementations')
plt.xlabel('z')
plt.ylabel('W(z)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower right', frameon=False)

plt.subplot(1, 2, 2)
plt.plot(z_values, errors, 'o', label='Absolute Error', color='green')
plt.title('Absolute Error Between Implementations')
plt.xlabel('z')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()
# Return the maximum error for a quick reference
max_error = np.max(errors)
max_error
print(errors)
