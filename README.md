# NonlinearHM
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15665163.svg)](https://doi.org/10.5281/zenodo.15665163)

This repository contains a 1D **Nonlinear Harmonic Model** and a **Linearized Harmonic Model**, developed to support a new interpretation framework for understanding nonlinear tidal dynamics.

## Contents

### 1. `Nonlinear_Harmonic_MarsdiepVlie.py`

This script implements a 1D nonlinear harmonic model based on the shallow water equations, incorporating **Defina’s wetting and drying approach**.

- The model simulates idealized tidal basins with varying **bathymetry** and **channel width**, controlled by parameters `α` (alpha) and `β` (beta), respectively.
- When `α = 1` and `β = 1`, the setup represents a simplified 1D configuration of the **Marsdiep–Vlie tidal inlet system** in the Dutch Wadden Sea.
- The provided initial conditions file:  
  `initial_distribution_variables_N30-alpha1-beta1.mat`  
  corresponds **only** to the `α = 1`, `β = 1` case.  
  Thus, for other scenarios, the initial conditions still need uploading, be careful about the loop of alpha-beta used in the code.
  **It is recommended that users run only the α = 1, β = 1 case as an example!**

### 2. `Linearized_Harmonic_MarsdiepVlie.py`

This script runs a **linearized** model using the output from the nonlinear model as input.

- It decomposes the results to assess the individual contributions of nonlinear forcing terms in the governing equations.

---

## Notes

- Please ensure MATLAB `.mat` files are readable by your Python environment (e.g., using `scipy.io.loadmat`).
- Both models are designed for **short, idealized runs**, making them lightweight and suitable for testing and educational purposes.

## Citation

If you use this model or dataset, please cite:

> Dong H. *NonlinearHM – 1D nonlinear and linearized tidal harmonic models*. Zenodo. [https://doi.org/10.5281/zenodo.15665163](https://doi.org/10.5281/zenodo.15665163)

