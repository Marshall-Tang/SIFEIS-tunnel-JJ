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

l_1 = np.array([0.5, 0.5, 1.5, 1.5]) * 1e-9  # insulator 1 thickness
l_2 = np.array([0.5, 0.5, 0.5, 0.5]) * 1e-9  # insulator 2 thickness

U_1 = np.array([0.15, 0.15, 0.15, 0.15]) * e  # insulator 1 barrier
U_2 = np.array([0.15, 0.4, 0.15, 0.4]) * e    # insulator 2 barrier

P_experiment = np.linspace(-8, 8, 51)        # polarization in μC/cm²
P = P_experiment * 0.01                      # polarization in C/m²
Gap_Sc = 2e-3 * e                            # superconducting gap ~2 meV

# ----------- Folders Setup -----------
RESULTS_FOLDER = Path("results/num_ana_compare")
DATA_FOLDER = RESULTS_FOLDER / "data"
FIG_FOLDER = RESULTS_FOLDER / "figures"

for folder in (RESULTS_FOLDER, DATA_FOLDER, FIG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------- Run Experiment -----------


def main():
    '''
    Compare numerical and analytical solutions for critical current density as a function of polarization.
    '''
    print("Running experiment 'num_anal_compare'...")
    t0 = time.time()
    # Generate and save data
    data_file_name_num = process.Data_polarization_dependence(
        Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER)
    data_file_name_analytical = process.Data_analytical_solution(
        Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER)
    data_file_name_analytical = DATA_FOLDER / "data_Jc_eta_P_analytical.npz"
    # Plot and save figures
    process.Plot_compare_num_analytical(
        data_file_name_analytical, data_file_name_num, FIG_FOLDER)

    dt = time.time() - t0
    print(f"Done. Running time: {dt:.2f} s")


if __name__ == "__main__":
    main()
