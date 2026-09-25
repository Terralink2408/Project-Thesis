import numpy as np
from scipy.signal import lfilter
from numpy.linalg import lstsq
from scipy import signal
from scipy.linalg import block_diag

import scipy as sp
import sympy as sm
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pysindy as ps
import joblib

import os
from pathlib import Path 

import sys

import control as control



plant_params = {
    "m": 1.0,
    "l": 0.5,
    "J": 0.25,
    "b": 0.05,
    "g": 9.81,
}

controller_params = {
    "K": np.array([-10.0, -2.0]),
    "theta_ref": 0,
    "u_max": 2.0
}



T = np.arange(0, 100, 0.01)


def system_dynamics(x, u, plant_params):
    J = plant_params["J"]
    m = plant_params["m"]
    g = plant_params["g"]
    l = plant_params["l"]
    b = plant_params["b"]

    theta, omega = x
    theta_dot = omega
    omega_dot = (m * g * l * np.sin(theta) - b * omega+ u) / J
    return np.array([theta_dot, omega_dot])

def reference_signal(t):
    return (
        np.deg2rad(10.0) * np.sin(0.8 * t)
    )

def reference_derivative(t):
    return (
        0.8 * np.deg2rad(10) * np.cos(0.8 * t)
    )


def controller(x, controller_params, theta_ref, omega_ref):
    K = controller_params["K"]
    u_max = controller_params["u_max"]

    theta_error = (x[0] - theta_ref + np.pi) % (2*np.pi) - np.pi
    omega_error = x[1] - omega_ref 

    error_state = np.array([theta_error, omega_error])

    u = float(K @ error_state)
    return np.clip(u, -u_max, u_max)


Ts = 0.01

# ============================================================
# Numerical integration
# ============================================================

def rk4_step(x, u, Ts, plant_params):
    k1 = system_dynamics(x, u, plant_params)
    k2 = system_dynamics(x + 0.5 * Ts * k1, u, plant_params)
    k3 = system_dynamics(x + 0.5 * Ts * k2, u, plant_params)
    k4 = system_dynamics(x + Ts * k3, u, plant_params)

    return x + Ts / 6.0 * (k1 + 2*k2 + 2*k3 + k4)



# ============================================================
# Simulation to generate data
# ============================================================

def simulate(x0, closed_loop):
    x_history = np.zeros((len(T), 2))
    u_history = np.zeros(len(T))

    x_history[0] = x0

    for k in range(len(T) - 1):
        x = x_history[k]



        if closed_loop:
            theta_ref = reference_signal(T[k])
            omega_ref = reference_derivative(T[k])
            u = controller(x, controller_params, theta_ref, omega_ref)
        else:
            u = 0.0

        u_history[k] = u

        x_history[k + 1] = rk4_step(
            x=x,
            u=u,
            Ts=Ts,
            plant_params=plant_params,
        )

    if closed_loop:
        theta_ref_final = reference_signal(T[-1])
        omega_ref_final = reference_derivative(T[-1])

        u_history[-1] = controller(
            x_history[-1],
            controller_params,
            theta_ref_final,
            omega_ref_final
        )

    return x_history, u_history


# ============================================================
# Run open- and closed-loop simulations
# ============================================================

x0 = np.array([
    np.deg2rad(0.0),  # initial angle
    0.0,              # initial angular velocity
])

x_open, u_open = simulate(x0, closed_loop=False)
x_closed, u_history = simulate(x0, closed_loop=True)

#saving history for VRFT
y_history = x_closed[:, 0].copy()
y_exp = y_history.copy()
u_exp = u_history.copy()



# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
)

reference_history = np.array([
    reference_signal(t) for t in T
])

# Angle
axes[0].plot(
    T,
    np.rad2deg(x_open[:, 0]),
    label="Open loop",
    linewidth=2,
)

axes[0].plot(
    T,
    np.rad2deg(x_closed[:, 0]),
    label="Closed loop",
    linewidth=2,
)

axes[0].plot(
    T,
    np.rad2deg(reference_history),
    label="Reference",
    color="black",
    linestyle="--",
    linewidth=2,
)

axes[0].set_ylabel(r"$\theta$ [degrees]")
axes[0].set_title("Nonlinear inverted pendulum")
axes[0].legend()
axes[0].grid(True)


# Angular velocity
axes[1].plot(
    T,
    x_open[:, 1],
    label="Open loop",
    linewidth=2,
)

axes[1].plot(
    T,
    x_closed[:, 1],
    label="Closed loop",
    linewidth=2,
)

axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[1].set_ylabel(r"$\omega$ [rad/s]")
axes[1].legend()
axes[1].grid(True)


# Control input
axes[2].plot(
    T,
    u_open,
    label="Open loop",
    linewidth=2,
)

axes[2].plot(
    T,
    u_history,
    label="Closed loop",
    linewidth=2,
)

axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[2].set_xlabel("T [s]")
axes[2].set_ylabel(r"$u$ [Nm]")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()



#============================================================
# VRFT stuff
# ============================================================

#filter parameters
tau = 0
q = 2
settling_time = 4

s = control.TransferFunction.s


#helper functions
def get_Coeff(system):
    num = np.asarray(system.num[0][0], dtype=float)
    den = np.asarray(system.den[0][0], dtype=float)

    num = np.pad(num, (len(den) - len(num), 0))
    num = num / den[0]
    den = den / den[0]
    return num, den

#Sanity check transfer function

K_theta, K_omega = controller_params["K"]
J = plant_params["J"]
m = plant_params["m"]
g = plant_params["g"]
l = plant_params["l"]
b = plant_params["b"]

M_continuous = control.tf(
    [-K_omega, -K_theta],
    [J, b - K_omega, -(m*g*l + K_theta)],
)

"""
a_0 = -10
a_1 = 0.5

M_continuous = (a_0*s + a_1) / (1 + 0.2*settling_time*s)**q #For now just setting tau to 0 to not worry about exponential in numerator
"""
M_d = control.sample_system(
    M_continuous,
    Ts,
    method="zoh",
)

y_exp = np.asarray(y_history, dtype=float).squeeze()
u_exp = np.asarray(u_history, dtype=float).squeeze()


F = M_d*(1-M_d)
F_aux = 1-M_d #For now, omitting frequency filters

F_num, F_den = get_Coeff(F)
F_auxNum, F_auxDen = get_Coeff(F_aux)



u_filtered = lfilter(F_num, F_den, u_exp, axis=-1, zi=None)
y_filtered = lfilter(F_num, F_den, y_exp, axis=-1, zi=None)
e_virtual = lfilter(F_auxNum, F_auxDen, y_exp, axis=-1, zi=None) - y_filtered # why not subttract y_exp

e_P = e_virtual

e_D = np.zeros_like(e_virtual)
e_D[1:] = np.diff(e_virtual) / Ts

Phi = np.column_stack([-e_P, -e_D])


theta, residuals, rank, singular_values = np.linalg.lstsq(
    Phi,
    u_filtered,
    rcond=None,
)

Kp_vrft, Kd_vrft = theta

print(f"Original PD parameters are {controller_params['K']}")
print(f"VRFT parameters are {theta[0]}, {theta[1]}")

# ============================================================
# Run open- and closed-loop simulations again with VRFT-parameters
# ============================================================

def reference_signal(t):
    return (
        np.deg2rad(10.0) * np.sin(0.8 * t)
    )

def reference_derivative(t):
    return (
        0.8 * np.deg2rad(10) * np.cos(0.8 * t)
    )

controller_params["K"] = theta

x0 = np.array([
    np.deg2rad(10.0),  # initial angle
    0.0,              # initial angular velocity
])

x_open, u_open = simulate(x0, closed_loop=False)
x_closed, u_history = simulate(x0, closed_loop=True)


#Need to pack the plotting into a neat function
# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
)

reference_history = np.array([
    reference_signal(t) for t in T
])

# Angle
axes[0].plot(
    T,
    np.rad2deg(x_open[:, 0]),
    label="Open loop",
    linewidth=2,
)

axes[0].plot(
    T,
    np.rad2deg(x_closed[:, 0]),
    label="Closed loop",
    linewidth=2,
)

axes[0].plot(
    T,
    np.rad2deg(reference_history),
    label="Reference",
    color="black",
    linestyle="--",
    linewidth=2,
)

axes[0].set_ylabel(r"$\theta$ [degrees]")
axes[0].set_title("Nonlinear inverted pendulum")
axes[0].legend()
axes[0].grid(True)

# Angular velocity
axes[1].plot(
    T,
    x_open[:, 1],
    label="Open loop",
    linewidth=2,
)

axes[1].plot(
    T,
    x_closed[:, 1],
    label="Closed loop",
    linewidth=2,
)

axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[1].set_ylabel(r"$\omega$ [rad/s]")
axes[1].legend()
axes[1].grid(True)


# Control input
axes[2].plot(
    T,
    u_open,
    label="Open loop",
    linewidth=2,
)

axes[2].plot(
    T,
    u_history,
    label="Closed loop",
    linewidth=2,
)

axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
axes[2].set_xlabel("T [s]")
axes[2].set_ylabel(r"$u$ [Nm]")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()