import numpy as np
from tqdm import tqdm
import src.func_SIFEIS as SIFEIS
from .plot import *
import time

plt.style.use(['./src/conf/prb.mplstyle'])


############ SIFEIS ##############

def Potential_profiles(p_value, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, FIG_folder):
    polarization_value = p_value

    for n, ll in enumerate(l_1):
        plot_data_x, plot_data_Up, plot_data_Um = SIFEIS.potential_profile(
            polarization_value, l_1[n], l_2[n], d, delta, U_0, U_1[n], U_2[n], rPerm_FE)
        plot_potential_profile(plot_data_x, plot_data_Up,
                               plot_data_Um, d, n, FIG_folder, show=False)
    return


def Data_polarization_dependence(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_folder):
    data_file_name = DATA_folder / "data_Jc_eta_P.npz"
    data_Jc_all = []
    data_efficiency_all = []
    for n, ll in enumerate(l_1):
        data_Jc = np.array([SIFEIS.critical_current_density_WKB(
            Ef, pp, l_1[n], l_2[n], d, delta, U_0, U_1[n], U_2[n], rPerm_FE, Gap_Sc) for pp in P])
        data_Jc_all.append(data_Jc)
        data_Jc_inverse_p = np.array([SIFEIS.critical_current_density_WKB(
            Ef, pp, l_1[n], l_2[n], d, delta, U_0, U_1[n], U_2[n], rPerm_FE, Gap_Sc) for pp in -P])
        data_efficiency = np.abs(
            (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
        data_efficiency_all.append(data_efficiency)
    data_P = P
    np.savez(data_file_name, Jc=data_Jc_all, eta=data_efficiency_all, P=data_P)

    return data_file_name


def Plot_polarization_dependence(data_file_name, FIG_folder):
    figure_1_file_name = FIG_folder / 'Jc_vs_Polarization.png'
    figure_2_file_name = FIG_folder / 'Efficiency_vs_Polarization.png'
    figure_3_file_name = FIG_folder / 'Jav_vs_Polarization.png'
    data = np.load(data_file_name)
    data_Jc = data['Jc']
    data_eta = data['eta']
    data_P = data['P']
    fig1, ax1 = plot_Jc_vs_polarization(data_P, data_Jc)
    fig2, ax2 = plot_efficiency_vs_polarization(data_P, data_eta)
    fig3, ax3 = plot_Jav_vs_polarization(data_P, data_Jc)
    fig1.savefig(figure_1_file_name)
    fig2.savefig(figure_2_file_name)
    fig3.savefig(figure_3_file_name)
    return figure_1_file_name, figure_2_file_name, figure_3_file_name


def Data_analytical_solution(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_folder):
    data_file_name = DATA_folder / "data_Jc_eta_P_analytical.npz"
    data_Jc_all = []
    data_efficiency_all = []
    for n, ll in enumerate(l_1):
        data_Jc = np.array([SIFEIS.critical_current_density_analytical(
            pp, ll, l_2[n], d, delta, U_0, U_1[n], U_2[n], rPerm_FE, Gap_Sc) for pp in P])
        data_Jc_all.append(data_Jc)
        data_Jc_inverse_p = np.array([SIFEIS.critical_current_density_analytical(
            pp, ll, l_2[n], d, delta, U_0, U_1[n], U_2[n], rPerm_FE, Gap_Sc) for pp in -P])
        data_efficiency = np.abs(
            (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
        data_efficiency_all.append(data_efficiency)
    data_P = P
    np.savez(data_file_name, Jc=data_Jc_all, eta=data_efficiency_all, P=data_P)

    return data_file_name


def Plot_compare_num_analytical(data_file_name_analytical, data_file_name_numerical, FIG_folder):
    figure_1_file_name = FIG_folder / 'Jc_vs_Polarization_compare.png'
    data_analytical = np.load(data_file_name_analytical)
    data_num = np.load(data_file_name_numerical)
    data_Jc_ana = data_analytical['Jc']
    # data_eta_ana = data_analytical['eta']
    data_Jc_num = data_num['Jc']
    # data_eta_num = data_num['eta']
    data_P = data_analytical['P']
    fig1, ax1 = plot_Jc_vs_polarization_compare_sep(
        data_P, data_Jc_ana, data_Jc_num)
    fig1.savefig(figure_1_file_name)
    return figure_1_file_name


def Plot_analytical_approximation_check(Ef, l_1, l_2, d, U_0, U_1, U_2, FIG_folder):
    figure_file_name = FIG_folder / 'analytical' / \
        'Analytical_approximation_check.png'
    'Analytical_approximation_check.png'
    k, exact_phase, approx_phase = SIFEIS.analytical_approximation_check(
        Ef, l_1, l_2, d, U_0, U_1, U_2)
    fig, ax = plot_analytical_approximation_check(k, exact_phase, approx_phase)
    fig.savefig(figure_file_name)
    return figure_file_name


def Data_thickness_dependence(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER):
    data_file_name = DATA_FOLDER / "_data_Jc_eta_thickness.npz"
    data_Jc = np.zeros((len(l_2), len(l_1)))
    data_Jc_inverse_p = np.zeros((len(l_2), len(l_1)))
    for i, ll_1 in enumerate(l_1):
        for j, ll_2 in enumerate(l_2):
            data_Jc[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, P, ll_1, ll_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc)
            data_Jc_inverse_p[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, -P, ll_1, ll_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc)
    data_efficiency = np.abs(
        (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
    np.savez(data_file_name, Jc=data_Jc, eta=data_efficiency, l_1=l_1, l_2=l_2)
    return data_file_name


def Plot_thickness_dependence(data_file_name, FIG_FOLDER):
    figure_file_name = FIG_FOLDER / 'Efficiency_vs_Thickness.png'
    data = np.load(data_file_name)
    data_eta = data['eta']
    data_l_1 = data['l_1']
    data_l_2 = data['l_2']
    fig, ax = plot_efficiency_vs_thicknesses_contour(
        data_eta, data_l_1, data_l_2)
    fig.savefig(figure_file_name)
    return figure_file_name


def Data_barrier_dependence(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_folder):
    data_file_name = DATA_folder / "data_Jc_eta_barrier.npz"
    data_Jc = np.zeros((len(U_2), len(U_1)))
    data_Jc_inverse_p = np.zeros((len(U_2), len(U_1)))
    for i, uu_1 in enumerate(U_1):
        for j, uu_2 in enumerate(U_2):
            data_Jc[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, P, l_1, l_2, d, delta, U_0, uu_1, uu_2, rPerm_FE, Gap_Sc)
            data_Jc_inverse_p[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, -P, l_1, l_2, d, delta, U_0, uu_1, uu_2, rPerm_FE, Gap_Sc)
            print(
                f"data_Jc[{j},{i}]={data_Jc[j,i]}, data_Jc_inverse_p[{j},{i}]={data_Jc_inverse_p[j,i]}")
    data_efficiency = np.abs(
        (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
    np.savez(data_file_name, Jc=data_Jc, eta=data_efficiency, U_1=U_1, U_2=U_2)
    return data_file_name


def Plot_barrier_dependence(data_file_name, FIG_folder):
    figure_file_name = FIG_folder / 'Efficiency_vs_Barriers.png'
    data = np.load(data_file_name)
    data_eta = data['eta']
    data_U_1 = data['U_1']
    data_U_2 = data['U_2']
    fig, ax = plot_efficiency_vs_barriers_contour(
        data_eta, data_U_1, data_U_2)
    fig.savefig(figure_file_name)
    return figure_file_name


def Data_thickness_barrier_dependence(Ef, P, L_tot, d, delta, U_0, U_tot, rPerm_FE, Gap_Sc, DATA_folder):
    data_file_name = DATA_folder / "data_Jc_eta_thickness_barrier.npz"
    l_1 = np.linspace(0, L_tot, 51)
    l_2 = L_tot-l_1
    U_1 = np.linspace(0.1*e, U_tot-0.1*e, 51)
    U_2 = U_tot-U_1
    data_dl = l_2 - l_1  # gives dl = l_2 - l_1
    data_dU = U_2 - U_1  # gives dl = U_2 - U_1
    data_Jc = np.zeros((len(U_1), len(l_1)))
    data_Jc_inverse_p = np.zeros((len(U_1), len(l_1)))
    for i, ll_1 in enumerate(l_1):
        for j, uu_1 in enumerate(U_1):
            data_Jc[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, P, l_1[i], l_2[i], d, delta, U_0, U_1[j], U_2[j], rPerm_FE, Gap_Sc)
            data_Jc_inverse_p[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, -P, l_1[i], l_2[i], d, delta, U_0, U_1[j], U_2[j], rPerm_FE, Gap_Sc)
    data_efficiency = np.abs(
        (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
    np.savez(data_file_name, Jc=data_Jc,
             eta=data_efficiency, dl=data_dl, dU=data_dU)
    return data_file_name


def Plot_thickness_barrier_dependence(data_file_name, FIG_folder):
    figure_1_file_name = FIG_folder / \
        'Efficiency_vs_ThickDiff_PotentialDiff_contour.png'
    figure_2_file_name = FIG_folder / 'Efficiency_vs_ThickDiff_PotentialDiff.png'
    data = np.load(data_file_name)
    data_eta = data['eta']
    data_dl = data['dl']
    data_dU = data['dU']
    selected_levels = [0, 0.75 * np.max(data_dU), 0.75 * np.min(data_dU)]
    selected_indices = [np.argmin(np.abs(data_dU - selected_level)
                                  )for selected_level in selected_levels]
    print(selected_indices)
    selected_data_dU = data_dU[selected_indices]
    selected_data_eta = data_eta[selected_indices, :]
    fig1, ax1 = plot_efficiency_countour(data_eta, data_dl, data_dU)
    fig2, ax2 = plot_efficiency_vs_thickness(selected_data_eta, data_dl)
    fig1.savefig(figure_1_file_name)
    fig2.savefig(figure_2_file_name)
    return figure_1_file_name, figure_2_file_name


def Data_d_dependence(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_FOLDER):
    data_file_name = DATA_FOLDER / "data_Jc_eta_d.npz"
    data_Jc = np.zeros((len(rPerm_FE), len(d)))
    data_Jc_inverse_p = np.zeros((len(rPerm_FE), len(d)))
    for j, rp in enumerate(rPerm_FE):
        for i, dd in enumerate(d):
            # V_tilt = e * P * dd * delta / Perm_0 / (2 * rp * delta + dd)
            # print(f"U_0 - |V_tilt| = {U_0/e - np.abs(V_tilt/e)} eV")
            data_Jc[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, P, l_1, l_2, dd, delta, U_0, U_1, U_2, rp, Gap_Sc)
            data_Jc_inverse_p[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, -P, l_1, l_2, dd, delta, U_0, U_1, U_2, rp, Gap_Sc)
    data_efficiency = np.abs(
        (data_Jc - data_Jc_inverse_p) / (data_Jc + data_Jc_inverse_p))
    np.savez(data_file_name, Jc=data_Jc,
             eta=data_efficiency, d=d, rPerm_FE=rPerm_FE)
    return data_file_name


def Plot_d_dependence(data_file_name, FIG_folder):
    figure_file_name = FIG_folder / 'Efficiency_vs_d.png'
    data = np.load(data_file_name)
    data_eta = data['eta']
    data_d = data['d']
    data_rPerm = data['rPerm_FE']
    fig, ax = plot_efficiency_vs_d(data_eta, data_d, data_rPerm)
    fig.savefig(figure_file_name)
    return figure_file_name


def Data_material_dependence(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, DATA_folder):
    data_file_name = DATA_folder / "data_eta_material_log.npz"
    data_Jc = np.zeros((len(rPerm_FE), len(delta)))
    data_Jc_inverse_p = np.zeros((len(rPerm_FE), len(delta)))
    for i, dt in enumerate(tqdm(delta, desc="Processing delta")):
        # for i, dt in enumerate(delta):
        for j, rp in enumerate(tqdm(rPerm_FE, desc="Processing rPerm_FE")):
            # for j, rp in enumerate(rPerm_FE):
            data_Jc[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, P, l_1, l_2, d, dt, U_0, U_1, U_2, rp, Gap_Sc)
            data_Jc_inverse_p[j, i] = SIFEIS.critical_current_density_WKB(
                Ef, -P, l_1, l_2, d, dt, U_0, U_1, U_2, rp, Gap_Sc)
    data_efficiency = np.abs(
        (data_Jc - data_Jc_inverse_p) / np.abs(data_Jc + data_Jc_inverse_p))
    np.savez(data_file_name, Jc=data_Jc,
             eta=data_efficiency, delta=delta, rPerm_FE=rPerm_FE)
    return data_file_name


def Plot_material_dependence(data_file_name, FIG_folder):
    figure_file_name = FIG_folder / 'Efficiency_vs_material_log.png'
    data = np.load(data_file_name)
    data_eta = data['eta']
    data_delta = data['delta']
    data_rPerm_FE = data['rPerm_FE']
    fig, ax = plot_efficiency_vs_material_contour(
        data_eta, data_delta, data_rPerm_FE)
    fig.savefig(figure_file_name)
    return figure_file_name


# def Efficiency_vs_d(E, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, FIG_folder):
#     plot_data_efficiency_all = []
#     for n, rp in enumerate(rPerm_FE):
#         plot_data_efficiency = SIFEIS.efficiency_WKB(
#             E, P, d, l_1, l_2, delta, U_0, U_1, U_2, rp, Gap_Sc)
#         plot_data_efficiency_all.append(plot_data_efficiency)
#     plot_data_d = d
#     plot_data_rPerm = rPerm_FE
#     return plot_efficiency_vs_d(plot_data_efficiency_all, plot_data_d, plot_data_rPerm, FIG_folder, show=False)


# def Efficiency_vs_ThickDiff_PotentialDiff_contour(E, P, L_tot, d, delta, U_0, U_tot, rPerm_FE, Gap_Sc, FIG_folder):
#     l_1 = np.linspace(0, L_tot, 51)
#     l_2 = L_tot-l_1
#     U_1 = np.linspace(0.1*e, U_tot-0.1*e, 51)
#     U_2 = U_tot-U_1
#     plot_data_delta_l = l_2 - l_1  # gives l_2 - l_1
#     plot_data_delta_U = U_2 - U_1  # gives U_2 - U_1
#     plot_data_delta_l, plot_data_delta_U = np.meshgrid(
#         plot_data_delta_l, plot_data_delta_U)
#     l_1_mesh = (L_tot-plot_data_delta_l)/2
#     l_2_mesh = (L_tot+plot_data_delta_l)/2
#     U_1_mesh = (U_tot-plot_data_delta_U)/2
#     U_2_mesh = (U_tot+plot_data_delta_U)/2
#     plot_data_efficiency = SIFEIS.efficiency_WKB(
#         E, P, d, l_1_mesh, l_2_mesh, delta, U_0, U_1_mesh, U_2_mesh, rPerm_FE, Gap_Sc)
#     return plot_efficiency_countour(plot_data_efficiency, plot_data_delta_l, plot_data_delta_U, P, d, delta, l_1, U_0, U_1, FIG_folder, show=False)


# def Efficiency_vs_ThickDiff_PotentialDiff(E, P, L_tot, d, delta, U_0, U_tot, rPerm_FE, Gap_Sc, FIG_folder):
#     l_1 = np.linspace(0, L_tot, 51)
#     l_2 = L_tot-l_1
#     U_1 = np.array([U_tot/2, U_tot/5, U_tot*4/5])
#     U_2 = U_tot-U_1
#     plot_data_delta_l = l_2 - l_1  # gives l_2 - l_1
#     U_diff = U_2 - U_1  # gives U_2 - U_1
#     plot_data_efficiency_all = []
#     for n, u_d in enumerate(U_diff):
#         plot_data_efficiency = SIFEIS.efficiency_WKB(
#             E, P, d, l_1, l_2, delta, U_0, U_1[n], U_2[n], rPerm_FE, Gap_Sc)
#         plot_data_efficiency_all.append(plot_data_efficiency)
#     return plot_efficiency_vs_thickness(plot_data_efficiency_all, plot_data_delta_l, U_diff, P, d, delta, l_1, U_0, U_1, FIG_folder, show=False)


# def Efficiency_vs_d(E, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc, FIG_folder):
#     plot_data_efficiency_all = []
#     for n, rp in enumerate(rPerm_FE):
#         plot_data_efficiency = SIFEIS.efficiency_WKB(
#             E, P, d, l_1, l_2, delta, U_0, U_1, U_2, rp, Gap_Sc)
#         plot_data_efficiency_all.append(plot_data_efficiency)
#     plot_data_d = d
#     plot_data_rPerm = rPerm_FE
#     return plot_efficiency_vs_d(plot_data_efficiency_all, plot_data_d, plot_data_rPerm, FIG_folder, show=False)
