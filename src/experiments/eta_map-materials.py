import time
from pathlib import Path

import numpy as np

from src import process
from src.conf.CONSTANTS import e


# ----------- Parameters Setup -----------
Ef = 5.0 * e                                # Fermi energy of the superconductor
delta = np.logspace(-2, 2, 101) * 1e-9      # Fermi screening length
rPerm_FE = np.logspace(1, 3, 101)           # relative permittivity of FE
d = 2 * 1e-9                                # FE thickness
U_0 = 0.1 * e                               # FE potential barrier

l_1 = 1.5e-9          # insulator 1 thickness
l_2 = 0.5e-9          # insulator 2 thickness

U_1 = 0.15 * e       # insulator 1 barrier
U_2 = 0.4 * e        # insulator 2 barrier


P_experiment = 5            # polarization in μC/cm²
P = P_experiment * 0.01     # polarization in C/m²
Gap_Sc = 2e-3 * e           # superconducting gap ~2 meV

# ----------- Folders Setup -----------
RESULTS_FOLDER = Path("results/_eta_map-materials")
DATA_FOLDER = RESULTS_FOLDER / "data"
FIG_FOLDER = RESULTS_FOLDER / "figures"

for folder in (RESULTS_FOLDER, DATA_FOLDER, FIG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------- Run Experiment -----------


def main():
    '''
    Simulate the efficiency (eta) as a function of the dielectric constant (rPerm_FE) and screening length (delta).
    '''
    # np.seterr(divide='raise', invalid='raise')
    print("Running experiment 'eta_map-materials'...")
    t0 = time.time()
    # Generate and save data
    data_file_name = process.Data_material_dependence(
        Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER)
    # Plot and save figures
    process.Plot_material_dependence(data_file_name, FIG_FOLDER)

    dt = time.time() - t0
    print(f"Done. Running time: {dt:.2f} s")


if __name__ == "__main__":
    main()
