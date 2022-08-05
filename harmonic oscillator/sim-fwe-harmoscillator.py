"""
Forward Euler-based simulation for a driven damped harmonic oscillator.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv


# System variables
m = 2.0 # mass
k = 2.0 # spring stiffness
c = 0.0 # damping (critical = 2\sqrt(mk) = 4.0)

F0 = 1.0 # amplitude of forcing term
dt = 0.001 # time step
omega = 1.0 # frequency of forcing term
time = np.arange(0.0, 20.0, dt)
T = len(time)

# Allocate
y_ = np.zeros((2,T))
F_ = np.zeros((2,T))

# Initial state
y_0 = np.array([0,1])

# Dynamics matrices
A = np.array([[m,0],[ 0,1]])
B = np.array([[c,k],[-1,0]])

# Define previous timepoint
y_tmin1 = y_0

# Time-stepping
for (ii,t) in enumerate(time):

    # Compute current forcing term
    F_[0,ii] = F0*np.cos(omega*t)

    # Update state
    y_[:,ii] = y_tmin1 + dt*inv(A)@(F_[:,ii] - B@y_tmin1)

    # Update previous variable
    y_tmin1 = y_[:,ii]

    # Compute energies
    KE = 0.5*m*y_[0,ii]**2
    PE = 0.5*k*y_[1,ii]**2

    if t % 1 <= 0.01:
        print('Total energy:', KE+PE)

print('Critical damping:', np.sqrt( (-c**2 + 4*m*k)/(2.*m) ))
print('Natural frequency:', np.sqrt(k/m))

# Plotting
plt.plot(time, y_[1,:], label="displacement")
# plt.plot(time, y_[0,:], label="velocity")
plt.plot(time, F_[0,:], label="force")
plt.grid(True)
plt.show()