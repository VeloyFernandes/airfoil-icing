"""Data from the 2024 group report.

"CFD and Experimental Analysis of the Ice Accretion Effects on Wind Turbine
Performance", FPS-1, Aalborg University Esbjerg, December 2024.
A. Kyawo, D. Arslanbek, V. Fernandes, Z. Bohlooli Zanjani; supervisor M. Mandø.
NACA 2412, chord 0.09 m, span 0.25 m, 20 m/s.
"""
import numpy as np

CHORD, SPAN, SPEED = 0.09, 0.25, 20.0
TUNNEL_SECTION = 0.305 * 0.305           # m^2
ICING = dict(mvd=20e-6, lwc=1.0e-3, temperature=248.15, duration=420.0)
FENSAP_ICE_MASS = 0.01024                # kg at 0 deg after 420 s

ALPHA = np.arange(9.0)
# Ansys Fluent, SST k-omega, extruded section (Tables 5.5 and 5.8)
CFD_CLEAN_CL = np.array([0.16509, 0.26441, 0.36273, 0.45913, 0.55127, 0.63949, 0.72179, 0.79613, 0.85597])
CFD_CLEAN_CD = np.array([0.0214, 0.021664, 0.022263, 0.023252, 0.024602, 0.026513, 0.029024, 0.032373, 0.037139])
CFD_ICED_CL = np.array([0.15077, 0.23394, 0.3175, 0.40108, 0.47954, 0.54645, 0.60781, 0.65241, 0.44777])
CFD_ICED_CD = np.array([0.034029, 0.036219, 0.036033, 0.042627, 0.037259, 0.042137, 0.046239, 0.053703, 0.088099])
# AAU Esbjerg wind tunnel (Tables 5.6 and 5.9)
EXP_CLEAN_CL = np.array([0.16326, 0.23596, 0.30951, 0.36589, 0.4313, 0.50022, 0.58332, 0.66229, 0.7314])
EXP_CLEAN_CD = np.array([0.013865, 0.018875, 0.024983, 0.024674, 0.02984, 0.035778, 0.039552, 0.045402, 0.054882])
EXP_ICED_ALPHA = np.array([0.0, 4.0, 8.0])
EXP_ICED_CL = np.array([0.10599, 0.4143, 0.71267])
EXP_ICED_CD = np.array([0.015442, 0.038398, 0.073688])

# Mesh study at 0 deg: coarse, medium, fine (224 720, 450 000, 898 880 cells; r = sqrt(2))
MESH_CL = (0.155, 0.165, 0.170)
MESH_CD = (0.023, 0.021, 0.021)
MESH_RATIO = 2 ** 0.5
