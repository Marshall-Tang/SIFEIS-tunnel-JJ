import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from .conf.CONSTANTS import *  # import constants
# import scienceplots

# plt.style.use(['science','nature'])
plt.style.use(['./src/conf/prb.mplstyle'])


def plot_potential_profile(x, y_p, y_m, d, index, fn=None, show=False):
    fig, ax = plt.subplots(figsize=(1.7, 0.85))
    d = d*1e9
    x = x*1e9
    y_p = y_p/e
    y_m = y_m/e
    ax.plot(x, y_p, linestyle='-', linewidth=1.5, color='#0000a2')
    ax.plot(x, y_m, linestyle='dashed', linewidth=1.5, color='#bc272d')

    # Fill from x.min() to 0
    ax.fill_between(x[(x >= x.min()) & (x <= 0)], 0, y_p[(
        x >= x.min()) & (x <= 0)], alpha=0.3, color='#50ad9f', linewidth=0)
    ax.fill_between(x[(x >= x.min()) & (x <= 0)], 0, y_m[(
        x >= x.min()) & (x <= 0)], alpha=0.3, color='#50ad9f', linewidth=0)

    # Fill from d to x.max()
    ax.fill_between(x[(x >= d) & (x <= x.max())], 0, y_p[(
        x >= 2) & (x <= x.max())], alpha=0.3, color='#50ad9f', linewidth=0)
    ax.fill_between(x[(x >= d) & (x <= x.max())], 0, y_m[(
        x >= 2) & (x <= x.max())], alpha=0.3, color='#50ad9f', linewidth=0)

    # Fill from 0 to 2
    ax.fill_between(x[(x >= 0) & (x <= 2)], 0, y_p[(
        x >= 0) & (x <= 2)], alpha=0.5, color='#e9c716', linewidth=0)
    ax.fill_between(x[(x >= 0) & (x <= 2)], 0, y_m[(
        x >= 0) & (x <= 2)], alpha=0.5, color='#e9c716', linewidth=0)

    x_ticks = [x.min(), 0, 2, x.max()]
    x_labels = [f"{x.min():.1f}", "0", "2", f"{x.max():.1f}"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(major_locator)
    ax.tick_params(axis='x', which='both', bottom=False,
                   top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False,
                   right=False, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    # Add padding between y-axis and plot area
    ax.set_xlim(left=x.min() - 0.3)  # Add 0.5 nm padding to the left
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('U (eV)')

    if fn is not None:
        if not os.path.exists(fn):
            os.makedirs(fn)
        fn = os.path.join(fn, f'potential_profile_{index}')
        plt.savefig(fn)
    if show:
        plt.show()
    return fig, ax


# def _plot_Jc_vs_polarization_combined(J_c, eta, P, x, y_p, y_m, l_1, l_2, d, U_0, U_1, U_2, fn=None, show=False):
#     fig, ax1 = plt.subplots(figsize=(2.4, 2))
#     fig.tight_layout()
#     J_c = J_c * 1e-3  # convert the unit to nA/um2
#     ax1.plot(P*100, J_c, linestyle='-', color='black', linewidth=1.5)
#     ax1.set_xlabel('P' + r' $(\mu C/cm^2)$')
#     ax1.set_ylabel(r'$J_c$' + r' $(nA/\mu m^2)$')
#     ax2 = ax1.twinx()
#     ax2.plot(100*P[np.where(P >= 0)], eta[np.where(P >= 0)],
#              linestyle='--', color='black', linewidth=1.5)
#     ax2.set_ylim(-0.1, 1)
#     ax2.set_ylabel(r'$\eta$')
#     axin = ax1.inset_axes([0.32, 0.55, 0.4, 0.4])
#     d = d*1e9
#     x = x*1e9
#     y_p = y_p/e
#     y_m = y_m/e
#     axin.plot(x, y_p, linestyle='-', linewidth=1.5, color='blue')
#     axin.plot(x, y_m, linestyle='--', linewidth=1.5, color='red')
#     axin.axvline(x=0, ymin=0, ymax=1, color='grey',
#                  linestyle='--', linewidth=1, alpha=0.5)
#     axin.axvline(x=d, ymin=0, ymax=1, color='grey',
#                  linestyle='--', linewidth=1, alpha=0.5)
#     axin.set_ylim([0, None])
#     axin.set_xlabel('x (nm)')
#     axin.set_ylabel('U (eV)')
#     axin.tick_params(axis='both', which='major', pad=2)
#     if fn is not None:
#         if not os.path.exists(fn):
#             os.makedirs(fn)
#         fn = os.path.join(fn, f'Jc-P_profile_combined.png')
#         plt.savefig(fn, dpi=600, bbox_inches='tight')
#     if show:
#         plt.show()
#     return fig, ax1


def plot_Jc_vs_polarization(P, J_c):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    colors = ['black', '#0000a2', '#bc272d', '#e9c716']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for n, j_c in enumerate(J_c):
        ax.plot(P*100, j_c*1e-3, color=colors[n],
                linestyle=linestyles[n], linewidth=1.2)
    ax.set_xlabel('P' + r' $(\mu C/cm^2)$')
    ax.set_ylabel(r'$J_c$' + r' $(nA/\mu m^2)$')
    ax.set_yscale('log')
    return fig, ax


def plot_Jav_vs_polarization(P, J_c):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    colors = ['black', '#0000a2', '#bc272d', '#e9c716']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for n, j_c in enumerate(J_c):
        # Calculate Jav as the average of J_c for positive and negative P
        j_av = (j_c + j_c[::-1]) / 2
        # Only plot for P > 0
        positive_indices = P >= 0
        ax.plot(P[positive_indices]*100, j_av[positive_indices]*1e-3, color=colors[n],
                linestyle=linestyles[n], linewidth=1.2)
    ax.set_xlabel('P (µC/cm²)')
    ax.set_ylabel(r'$\bar{J_c}$ (nA/µm²)')
    ax.set_yscale('log')
    return fig, ax


def plot_efficiency_vs_polarization(P, efficiency):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    colors = ['black', '#0000a2', '#bc272d', '#e9c716']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for n, eta in enumerate(efficiency):
        positive_indices = P >= 0
        ax.plot(P[positive_indices]*100, eta[positive_indices], color=colors[n],
                linestyle=linestyles[n], linewidth=1.2)
    ax.set_xlabel('P (µC/cm²)')
    ax.set_ylabel(r'$\eta$')
    ax.set_ylim(-0.1, 1)
    return fig, ax


def plot_Jc_vs_polarization_compare_sep(P, J_c_ana, J_c_num):
    fig, axes = plt.subplots(4, 1, figsize=(
        2.7, 3), sharex=True, gridspec_kw={'hspace': 0.1})
    colors = ['black', '#0000a2', '#bc272d', '#e9c716']
    for n, ax in enumerate(axes):
        ax.plot(P*100, J_c_num[n]*1e-3, color=colors[n],
                linestyle='-', linewidth=1.2, label='Numerical')
        ax.plot(P*100, J_c_ana[n]*1e-3, color=colors[n],
                linestyle='--', linewidth=1.2, label='Analytical')
        # ax.set_ylabel(r'$J_c$', fontsize=8)
        # ax.legend(loc='upper right', fontsize=6, frameon=False)
        # if n < 3:
        #     ax.tick_params(labelbottom=False)
    axes[0].set_ylim(250, 500)
    axes[1].set_ylim(70, 180)
    axes[2].set_ylim(top=25)
    axes[3].set_ylim(top=9)
    axes[0].yaxis.set_major_locator(MultipleLocator(100))
    axes[1].yaxis.set_major_locator(MultipleLocator(50))
    axes[2].yaxis.set_major_locator(MultipleLocator(10))
    axes[3].yaxis.set_major_locator(MultipleLocator(5))
    axes[-1].set_xlabel('P (µC/cm²)')
    fig.text(-0.2, 0.5, r'$J_c$ (nA/µm²)',
             va='center', rotation='vertical')
    return fig, axes


def plot_analytical_approximation_check(k, data_exact, data_approx):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    ax.plot(k, data_exact, linestyle='-', color='black',
            linewidth=1.2, label='Exact Phase')
    ax.plot(k, data_approx, linestyle='--', color='black',
            linewidth=1.2, label='Approximate Phase')
    ax.set_xlabel('k')
    ax.set_ylabel('k*T')
    ax.legend(loc='upper right', fontsize=6, frameon=False)
    return fig, ax


# def plot_WKB_phase(P, gamma_0, gamma_1, gamma_2, l_1, l_2, d, U_0, U_1, U_2, fn=None, show=False):
#     fig, ax = plt.subplots()
#     gamma = gamma_0+gamma_1+gamma_2
#     ax.plot(P, gamma, linestyle='-', linewidth=2, color='red')
#     # ax.plot(P,gamma_0,linestyle='--',linewidth=2,color='orange')
#     # ax.plot(P,gamma_1,linestyle='--',linewidth=2,color='green')
#     # ax.plot(P,gamma_2,linestyle='--',linewidth=2,color='blue')
#     if fn is not None:
#         if not os.path.exists(fn):
#             os.makedirs(fn)
#         fn = os.path.join(fn, f'Phase-P_relation.png')
#         plt.savefig(fn, dpi=600, bbox_inches='tight')
#     if show:
#         plt.show()
#     return fig, ax


# def plot_Jc_countour(Jc, delta_l, delta_U):
#     fig, ax = plt.subplots(figsize=(1.4, 1.4))
#     contourf = ax.contourf(delta_l*1e9, delta_U/e, Jc*1e-3, norm=LogNorm())
#     # contour = ax.contour(
#     #     contourf, levels=[0.3, 0.6, 0.8], colors='k', linestyles='--', linewidths=0.5)
#     # ax.clabel(contour, inline=True, fontsize=8)
#     plt.xlabel(r'$l_2-l_1$'+' (nm)')
#     plt.ylabel(r'$U_2-U_1$'+' (eV)')
#     cbar_ax = fig.add_axes([0.95, 0.122, 0.05, 0.7])
#     cbar = fig.colorbar(contourf, format="%0.2f", cax=cbar_ax)
#     cbar.ax.set_title(r'$J_c$')
#     return fig, ax


def plot_efficiency_countour(efficiency, delta_l, delta_U):
    fig, ax = plt.subplots(figsize=(1.4, 1.4))
    contourf = ax.contourf(delta_l*1e9, delta_U/e,
                           efficiency, levels=np.linspace(0, 1, 21))
    contour = ax.contour(
        contourf, levels=[0.3, 0.6, 0.8], colors='k', linestyles='--', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=1)
    ax.axhline(y=0.6, linestyle='dotted', color='#0000a2', linewidth=1)
    ax.axhline(y=-0.6, linestyle='dashed', color='#bc272d', linewidth=1)

    plt.xlabel(r'$l_2-l_1$'+' (nm)')
    plt.ylabel(r'$U_2-U_1$'+' (eV)')
    cbar_ax = fig.add_axes([0.95, 0.122, 0.05, 0.7])
    cbar = fig.colorbar(contourf, format="%0.2f", cax=cbar_ax)
    cbar.ax.set_title(r'$\eta$')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # if fn is not None:
    #     if not os.path.exists(fn):
    #         os.makedirs(fn)
    #     fn = os.path.join(fn, f'efficiency_phase_diagram')
    #     plt.savefig(fn)
    # if show:
    #     plt.show()
    return fig, ax


def plot_efficiency_vs_thickness(efficiency, delta_l):
    fig, ax = plt.subplots(figsize=(1.4, 1.4))
    colors = ['black', '#0000a2', '#bc272d']
    linestyles = ['solid', 'dotted', 'dashed']
    for n, eta in enumerate(efficiency):
        ax.plot(
            delta_l*1e9, efficiency[n], linestyle=linestyles[n], color=colors[n], linewidth=1.2)

    plt.xlabel(r'$l_2-l_1$'+' (nm)')
    plt.ylabel(r'$\eta$')
    ax.set_ylim(-0.1, 1)
    # if fn is not None:
    #     if not os.path.exists(fn):
    #         os.makedirs(fn)
    #     fn = os.path.join(fn, f'efficiency_vs_thickness')
    #     plt.savefig(fn)
    # if show:
    #     plt.show()
    return fig, ax


def plot_efficiency_vs_thicknesses_contour(efficiency, l_1, l_2):
    fig, ax = plt.subplots(figsize=(1.4, 1.4))
    contourf = ax.contourf(l_1*1e9, l_2*1e9, efficiency,
                           levels=np.linspace(0, 1.0, 21))
    contour = ax.contour(
        contourf, levels=[0.3, 0.6, 0.8], colors='k', linestyles='--', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    plt.xlabel(r'$l_1$'+' (nm)')
    plt.ylabel(r'$l_2$'+' (nm)')
    ax.set_aspect('equal')
    major_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_locator(major_locator)
    cbar_ax = fig.add_axes([0.95, 0.122, 0.05, 0.7])
    cbar = fig.colorbar(contourf, format="%0.2f", cax=cbar_ax)
    cbar.ax.set_title(r'$\eta$')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # if fn is not None:
    #     if not os.path.exists(fn):
    #         os.makedirs(fn)
    #     fn = os.path.join(fn, f'efficiency_thickness_contour')
    #     plt.savefig(fn)
    # if show:
    #     plt.show()
    return fig, ax


def plot_efficiency_vs_barriers_contour(efficiency, U_1, U_2, fn=None, show=False):
    fig, ax = plt.subplots(figsize=(1.4, 1.4))
    contourf = ax.contourf(U_1/e, U_2/e, efficiency,
                           levels=np.linspace(0, 1.0, 21))
    contour = ax.contour(
        contourf, levels=[0.1, 0.2, 0.3], colors='k', linestyles='--', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    plt.xlabel(r'$U_1$'+' (eV)')
    plt.ylabel(r'$U_2$'+' (eV)')
    ax.set_aspect('equal')
    major_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_locator(major_locator)
    cbar_ax = fig.add_axes([0.95, 0.122, 0.05, 0.7])
    cbar = fig.colorbar(contourf, format="%0.2f", cax=cbar_ax)
    cbar.ax.set_title(r'$\eta$')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if fn is not None:
        if not os.path.exists(fn):
            os.makedirs(fn)
        fn = os.path.join(fn, f'efficiency_barriers_contour')
        plt.savefig(fn)
    if show:
        plt.show()
    return fig, ax


def plot_efficiency_vs_d(efficiency, d, rPerm_FE):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    # colors = ['black', '#0000a2', '#bc272d', '#e9c716']
    colors = ['#e9c716', '#bc272d', '#0000a2', 'black']
    # linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    linestyles = ['dashdot', 'dashed', 'dotted', 'solid']
    for n, eta in enumerate(efficiency):
        ax.plot(d*1e9, eta, color=colors[n],
                linestyle=linestyles[n], linewidth=1.2)
    ax.legend([rf' $\varepsilon$={rPerm_FE[0]}', rf' $\varepsilon$={rPerm_FE[1]}', rf' $\varepsilon$={rPerm_FE[2]}',
              rf' $\varepsilon$={rPerm_FE[3]}'], loc=2, handletextpad=0.2, handlelength=1, framealpha=1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('d' + r' (nm)')
    ax.set_ylabel(r'$\eta$')
    ax.set_ylim(-0.1, 1)
    return fig, ax


def plot_efficiency_vs_material_contour(efficiency, delta, rPerm_FE):
    fig, ax = plt.subplots(figsize=(1.7, 1.7))
    contourf = ax.contourf(delta*1e9, rPerm_FE, efficiency,
                           levels=np.linspace(0, 1.0, 16))
    # contour = ax.contour(
    #     contourf, levels=[0.2, 0.4, 0.8], colors='k', linestyles='--', linewidths=0.5)
    # ax.clabel(contour, inline=True, fontsize=8)

    # Set log scale for both axes (ensure delta and df are > 0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    major_locator_x = mpl.ticker.LogLocator(base=10.0, numticks=10)
    ax.xaxis.set_major_locator(major_locator_x)
    minor_locator_x = mpl.ticker.LogLocator(
        base=10.0, subs=np.arange(2, 10)*0.1, numticks=10)
    ax.xaxis.set_minor_locator(minor_locator_x)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    major_locator_y = mpl.ticker.LogLocator(base=10.0, numticks=10)
    ax.yaxis.set_major_locator(major_locator_y)
    plt.xlabel(r'$\delta$'+' (nm)')
    plt.ylabel(r'$\varepsilon_{f}$')
    cbar_ax = fig.add_axes([0.95, 0.122, 0.05, 0.7])
    cbar = fig.colorbar(contourf, format="%0.2f", cax=cbar_ax)
    cbar.ax.set_title(r'$\eta$')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return fig, ax


def plot_efficiency_vs_potential(efficiency, delta_U, P, l_1, l_2, d, delta, U_0, fn=None, show=False):
    fig, ax = plt.subplots(figsize=(2, 2))
    # plt.axis('square')
    fig.tight_layout()
    colors = ['green', 'blue', 'red']
    styles = ['-', '', '']
    markers = ['', '^', '.']
    for i, (x, y) in enumerate(zip(delta_U, efficiency)):
        ax.plot(x/e, y, linewidth=1,
                color=colors[i], linestyle=styles[i], marker=markers[i], markersize=1.5)

    plt.xlabel(r'$U_2-U_1$'+' (eV)')
    plt.ylabel(r'$\eta$')
    ax.legend(['U=2eV', 'U=1eV', 'U=0.5eV'], loc=3,
              handletextpad=0.1, handlelength=1)
    if fn is not None:
        if not os.path.exists(fn):
            os.makedirs(fn)
        fn = os.path.join(fn, f'efficiency_potential.png')
        plt.savefig(fn, dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_efficiency_vs_thickdiff_barrierdiff(efficiency_1, efficiency_2, delta_l, delta_U, fn=None, show=False):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(2, 2))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0)
    colors = ['green', 'blue', 'red']
    styles = ['-', '', '']
    markers = ['', '^', '.']
    for i, (x, y) in enumerate(zip(delta_l, efficiency_1)):
        ax[0].plot(x*1e9, y, linewidth=1, color=colors[i],
                   linestyle=styles[i], marker=markers[i], markersize=1)
    for i, (x, y) in enumerate(zip(delta_U, efficiency_2)):
        ax[1].plot(x/e, y, linewidth=1, color=colors[i],
                   linestyle=styles[i], marker=markers[i], markersize=1)
    ax[0].set_xlabel(r'$l_2-l_1$'+' (nm)')
    ax[1].set_xlabel(r'$U_2-U_1$'+' (eV)')
    ax[0].set_ylabel(r'$\eta$')
    ax[0].legend(['L=4nm', 'L=3nm', 'L=2nm'])
    ax[1].legend()
    if fn is not None:
        if not os.path.exists(fn):
            os.makedirs(fn)
        fn = os.path.join(fn, f'efficiency_thick_barrier.png')
        plt.savefig(fn, dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax
