import time
from pathlib import Path

import numpy as np

from src import process
from src.conf.CONSTANTS import e


# ----------- Parameters Setup -----------
delta = 0.1e-9        # Fermi screening length
rPerm_FE = 100        # relative permittivity of FE
d = 2e-9              # FE thickness
U_0 = 0.1 * e         # FE potential barrier

l_1 = np.array([0.5, 0.5, 1.5, 1.5]) * 1e-9  # insulator 1 thickness
l_2 = np.array([0.5, 0.5, 0.5, 0.5]) * 1e-9  # insulator 2 thickness

U_1 = np.array([0.15, 0.15, 0.15, 0.15]) * e  # insulator 1 barrier
U_2 = np.array([0.15, 0.4, 0.15, 0.4]) * e    # insulator 2 barrier

P_experiment = 5                # polarization in μC/cm²
P = P_experiment * 0.01         # polarization in C/m²
Gap_Sc = 2e-3 * e               # superconducting gap ~2 meV

# ----------- Folders Setup -----------
RESULTS_FOLDER = Path("results/potential_profiles")
DATA_FOLDER = RESULTS_FOLDER / "data"
FIG_FOLDER = RESULTS_FOLDER / "figures"

for folder in (RESULTS_FOLDER, DATA_FOLDER, FIG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------- Run Experiment -----------


def main():
    print("Running experiment 'potential_profiles'...")
    t0 = time.time()
    # Plot and save figures
    process.Potential_profiles(
        P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, FIG_FOLDER)

    dt = time.time() - t0
    print(f"Done. Running time: {dt:.2f} s")


if __name__ == "__main__":
    main()
