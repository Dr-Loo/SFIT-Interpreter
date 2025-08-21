# run_sfit_program.py

from sfit_interpreter import SymbolicPhaseField
import numpy as np

# Initial condition: standing wave
def initial_ϕ(x, t):
    return np.sin(x) * np.cos(t)

# Transition condition: forced field change
def transition_ϕ(x, t):
    return np.cos(x) * np.sin(t)

# Setup
ϕ_field = SymbolicPhaseField(domain=[-1, 1], initial_condition=initial_ϕ)

# Phase 1: evolve until t = 5
ϕ_field.evolve(κ=1.0, dt=0.001, steps=5000)

# Transition: set new field
ϕ_field.set_field(transition_ϕ)

# Phase 2: evolve until t = 10
ϕ_field.evolve(κ=1.0, dt=0.001, steps=5000)

# Measure and plot
ϕ_field.plot_observable()
