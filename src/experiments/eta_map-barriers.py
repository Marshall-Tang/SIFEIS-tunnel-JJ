import time
from pathlib import Path

import numpy as np

from src import process
from src.conf.CONSTANTS import e


# ----------- Parameters Setup -----------
Ef = 5.0 * e          # Fermi energy of the superconductor
delta = 0.1e-9        # Fermi screening length
rPerm_FE = 100        # relative permittivity of FE
d = 2e-9              # FE thickness
U_0 = 0.1 * e         # FE potential barrier

L = 2.0e-9      # total thickness of the insulators
U = 1.0*e       # total potential barrier height of the insulators


P_experiment = 5            # polarization in μC/cm²
P = P_experiment * 0.01     # polarization in C/m²
Gap_Sc = 2e-3 * e           # superconducting gap ~2 meV

# ----------- Folders Setup -----------
RESULTS_FOLDER = Path("results/eta_map-barriers")
DATA_FOLDER = RESULTS_FOLDER / "data"
FIG_FOLDER = RESULTS_FOLDER / "figures"

for folder in (RESULTS_FOLDER, DATA_FOLDER, FIG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------- Run Experiment -----------


def main():
    '''
    Simulate the efficiency (eta) as a function of thickness differences (l1-l2) and barrier differences (U1-U2).
    '''
    print("Running experiment 'eta_map-barriers'...")
    t0 = time.time()
    # Generate and save data
    data_file_name = process.Data_thickness_barrier_dependence(
        Ef, P, L, d, delta, U_0, U, rPerm_FE, Gap_Sc, DATA_FOLDER)
    # Plot and save figures
    process.Plot_thickness_barrier_dependence(data_file_name, FIG_FOLDER)

    dt = time.time() - t0
    print(f"Done. Running time: {dt:.2f} s")


if __name__ == "__main__":
    main()
