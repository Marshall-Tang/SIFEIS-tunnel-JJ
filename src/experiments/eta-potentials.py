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

l_1 = 1 * 1e-9  # insulator 1 thickness
l_2 = 1 * 1e-9  # insulator 2 thickness

U_1 = np.linspace(0.1, 2.0, 51) * e  # insulator 1 barrier
U_2 = np.linspace(0.1, 2.0, 51) * e  # insulator 2 barrier

P_experiment = 5            # polarization in μC/cm²
P = P_experiment * 0.01     # polarization in C/m²
Gap_Sc = 2e-3 * e           # superconducting gap ~2 meV

# ----------- Folders Setup -----------
RESULTS_FOLDER = Path("results/eta_potential")
DATA_FOLDER = RESULTS_FOLDER / "data"
FIG_FOLDER = RESULTS_FOLDER / "figures"

for folder in (RESULTS_FOLDER, DATA_FOLDER, FIG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------- Run Experiment -----------


def main():
    '''
    Simulate the efficiency (eta) as a function of insulator potential barriers (U1 and U2).
    '''
    print("Running experiment 'eta-potentials'...")
    t0 = time.time()
    # Generate and save data
    data_file_name = process.Data_barrier_dependence(
        Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER)
    # Plot and save figures
    process.Plot_barrier_dependence(data_file_name, FIG_FOLDER)

    dt = time.time() - t0
    print(f"Done. Running time: {dt:.2f} s")


if __name__ == "__main__":
    main()
