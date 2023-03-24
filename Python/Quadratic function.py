# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:26:49 2023

@author: g361a609
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Define the function to be minimized
def f(x):
    return x**2 - 4*x + 3

# Find the minimum of the function using the minimize_scalar function
res = minimize_scalar(f)

# Print the minimum value and the location of the minimum
print("Minimum value:", res.fun)
print("Location of minimum:", res.x)

# plot the function
x = np.linspace(-2, 6, 100)
y = f(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function to minimize')
plt.show()

# Plot the function and the minimum location
x_vals = np.linspace(-2, 6, 100)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals)
plt.plot(res.x, res.fun, 'ro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Quadratic Function with Minimum')
plt.show()