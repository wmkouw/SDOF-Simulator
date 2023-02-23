"""
Forward Euler-based simulation for a general wave equation.
"""

import numpy as np
import matplotlib.pyplot as plt


# System variables
D = 1.0

# Time scales
dt = 0.01
time = np.arange(0.0, 100.0, dt)
T = len(time)

# Allocate
x_ = np.zeros((T,2)) # State

# Initial state
x_0 = np.array([0.0, 0.0])

# Define previous timepoint
x_tmin1 = x_0

# Time-stepping
for (ii,t) in enumerate(time):

    # Update prey population
    x_[:,ii] = x_tmin1 + dt*[x_tmin1[2], ]

    # Update previous variables
    x_tmin1 = x_[:,ii]


# Plotting
plt.plot(time, x_[1,:], label="state")
plt.legend()
plt.grid(True)
plt.show()