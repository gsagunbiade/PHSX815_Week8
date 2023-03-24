# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:27:23 2023

@author: Gbenga Agunbiade
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the true probability of the coin landing heads
true_p = 0.45

# Generate some data by flipping the coin a fixed number of times
num_flips = 150
data = np.random.binomial(n=1, p=true_p, size=num_flips)

# Define the likelihood function for the model
def likelihood(data, p):
    num_heads = np.sum(data)
    num_tails = len(data) - num_heads
    return p**num_heads * (1-p)**num_tails

# Define the prior distribution for the parameter p
def prior(p):
    if p < 0 or p > 1:
        return 0
    else:
        return 1

# Define the posterior distribution for the parameter p
def posterior(data, p):
    return likelihood(data, p) * prior(p)

# Plot the posterior distribution for a range of possible values of p
possible_p_values = np.linspace(0, 1, 151)
posterior_values = np.zeros_like(possible_p_values)
for i, p in enumerate(possible_p_values):
    posterior_values[i] = posterior(data, p)
posterior_values /= np.trapz(posterior_values, possible_p_values)
plt.plot(possible_p_values, posterior_values)
plt.xlabel('True Parameter Value')
plt.ylabel('Measured Value')
plt.show()