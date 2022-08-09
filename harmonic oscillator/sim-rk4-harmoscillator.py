"""
Runge-Kutta-based simulation for a driven damped harmonic oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv


# System variables
m = 2.0         # mass
k = 2.0         # spring stiffness
c = 0.0         # damping (critical = 2\sqrt(mk) = 4.0)
F0 = 0.0        # amplitude of forcing term

# Time scales
dt = 0.1        # time step
omega = 1.0     # frequency of forcing term
time = np.arange(0.0, 40.0, dt)
T = len(time)

# Allocate
y_ = np.zeros((2,T))
F_ = np.zeros((2,T))

# Initial state
y_0 = np.array([0,1])

# Dynamics matrices
A = np.array([[m,0],[ 0,1]])
B = np.array([[c,k],[-1,0]])
Ai = inv(A)

def F(t, stop=15):
    "Forcing term"
    if t <= stop:
        return np.array([F0*np.cos(omega*t), 0.0])
    else:
        return np.array([0.0, 0.0])

def G(y,t):
    "Differential equation"
    return Ai@(F(t) - B@y)

def RK1(y,t,dt):

    K1 = G(y,t)

    return dt*K1

def RK2(y,t,dt, method='midpoint'):

    if method == 'midpoint':
        K1 = G(y,t)
        K2 = G(y + K1*dt/2, t+dt/2)
        return dt*K2
    elif method == 'heun':
        K1 = G(y,t)
        K2 = G(y + K1*dt, t+dt)
        return dt*(K1+K2)/2
    else:
        raise Exception("Chosen 'method' unknown. Try 'midpoint' or 'heun'.")
    
def RK4(y,t,dt):

    K1 = G(y,t)
    K2 = G(y + K1*dt/2, t+dt/2)
    K3 = G(y + K2*dt/2, t+dt/2)
    K4 = G(y + K3*dt, t+dt)

    return dt*1/6*(K1 + 2*K2 + 2*K3 + K4)   

# Define previous timepoint
y_tmin1 = y_0

# Time-stepping
for (ii,t) in enumerate(time):

    # Update state
    y_[:,ii] = y_tmin1 + RK4(y_tmin1, t, dt)

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