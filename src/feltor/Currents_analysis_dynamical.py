# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:41:10 2021
ANALYSIS PROGRAM FOR CONSERVATION OF CURRENTS EQUATION IN FELTOR:
@author: raulgerru
"""

import netCDF4 as nc
import numpy as np
from scipy import fftpack
import json
import math
import matplotlib.pyplot as plt
import matplotlib.widgets
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import sys

sys.modules[__name__].__dict__.clear()



#fn = "/home/raulgerru/Desktop/PhD files/Research/FELTOR/SIMULATIONS/Diag_test_files/conservation_test_partial_gf_1+2+3_diag.nc"
#fn = "/home/raulgerru/Desktop/PhD files/Research/FELTOR/SIMULATIONS/Sheath tests/conservation_sheath_1+2+3_diag.nc"
#fn = "/home/raulgerru/Desktop/PhD files/Research/FELTOR/SIMULATIONS/Sheath tests/sheath_test_LCFS_025_0_diag.nc"
fn = "/home/raulgerru/Desktop/PhD files/Research/FELTOR/SIMULATIONS/Sheath tests/sheath_test_010_pen0_diag.nc"
ds = nc.Dataset(fn)
inputfile = ds.inputfile
inputfile_json = json.loads(inputfile)

# Physical constants
e = 1.60218e-19  # C
m_H = 1.6726219e-27  # Kg
m_e = 9.10938356e-31  # Kg
eps_0 = 8.85418781e-12  #
mu_0 = 1.25663706e-6  #

# INPUT PARAMETERS (Include by hand)
R_0 = 0.56  # m
B_0 = 0.4  # T
n0 = 1.5e19  # m^-3
m_i = 2  # m_H
T0 = 10  # eV
tau=inputfile_json['physical']['tau']
Ti = T0 * tau

Omega_0 = e * B_0 / (m_i * m_H)
C = e * n0 * Omega_0

rho = ds['rho'][:]
eta = ds['eta'][:]  # Poloidal direction (from 0 to 2pi)
t = ds['time'][:]
time = 1e3 * ds['time'][:] / Omega_0


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

def pol_shift(f):
    k=f
    trans = list(k)
    new = trans[-180:] + trans[:-180]
    data = np.stack(new)
    return data

def filter(F):
    dat = F
    for i in range(0, t.size-1):
        dat[i]=pol_shift(dat[i])
        fft = fftpack.fft2(dat[i])
        fft[23:1897] = 0 #26:1894
        dat[i] = fftpack.ifft2(fft).real
    return dat

def filter_image(data):
    dat=data
    dat = pol_shift(dat)
    fft = fftpack.fft2(dat)
    fft[26:1894] = 0
    dat = fftpack.ifft2(fft).real
    return dat


class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=t.size-3, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '',
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        self.slider.set_val(i)



def edge_plot(magnitude, title, cmin=None, cmax=None, axes=None ):
    if axes is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1 = axes
    #cmin = -0.05
    #cmax = 0.05
    p = plt.pcolor(rho[(rho > rho_min) & (rho < rho_max)], (eta - math.pi) / math.pi,
                   filter_image(magnitude[:, (rho > rho_min) & (rho < rho_max)]), cmap='jet',  vmin=cmin, vmax=cmax)#, shading='gouraud')
    ax1.axvline(x=1, color='k', linestyle='--')
    ax1.axhline(-0.5, color='w', linestyle='--')
    ax1.axhline(0, color='w', linestyle='--')
    ax1.axhline(0.5, color='w', linestyle='--')
    ax1.autoscale(enable=True)
    ax1.set_xlabel('$\\rho $')
    ylabels = ('X-DOWN', 'OMP', 'UP', 'HFS', 'X-DOWN')
    y_pos = [-1, -0.5, 0, 0.5, 1]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels)
    ax1.set_title(title)
    plt.colorbar()
    return p


def edge_animation_bar(magnitude, title, cmin=None, cmax=None ):
    m=filter(magnitude)
    fig, ax = plt.subplots()
    #cmin = -1
    #cmax = 1
    cax = ax.pcolor(rho, (eta - math.pi) / math.pi, m[1, :-1, :-1], cmap='jet',  vmin=cmin, vmax=cmax)

    def animate(i):
        cax.set_array(m[i + 1, :-1, :-1].flatten('C'))
    fig.colorbar(cax)
    plt.axvline(x=1, color='k', linestyle='--')
    plt.axhline(-0.5, color='w', linestyle='--')
    plt.axhline(0, color='w', linestyle='--')
    plt.axhline(0.5, color='w', linestyle='--')
    plt.xlim(rho_min,  rho_max)
    plt.xlabel('$\\rho $')
    #plt.xlim([rho_min, rho_max])
    ylabels = ('X-DOWN', 'OMP', 'UP', 'HFS', 'X-DOWN')
    y_pos = [-1, -0.5, 0, 0.5, 1]
    plt.yticks(y_pos, ylabels)
    plt.title(title)
    anim = Player(fig, animate)

    plt.show()


def edge_animation_bar_2(magnitude1, title1, magnitude2, title2):
        cmin=-0.02
        cmax=0.05
        m1=filter(magnitude1)
        m2=filter(magnitude2)
        fig=plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(1,2,1)
        cax1 = ax1.pcolor(rho, (eta - math.pi) / math.pi, m1[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)
        ax2 = fig.add_subplot(1, 2, 2)
        cax2 = ax2.pcolor(rho, (eta - math.pi) / math.pi, m2[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)

        def animate(i):
            cax1.set_array(m1[i + 1, :-1, :-1].flatten('C'))
            cax2.set_array(m2[i + 1, :-1, :-1].flatten('C'))

        fig.colorbar(cax1)
        #ax2.colorbar(cax2)
        ax1.axvline(x=1, color='k', linestyle='--')
        ax1.axhline(-0.5, color='w', linestyle='--')
        ax1.axhline(0, color='w', linestyle='--')
        ax1.axhline(0.5, color='w', linestyle='--')
        ax2.axvline(x=1, color='k', linestyle='--')
        ax2.axhline(-0.5, color='w', linestyle='--')
        ax2.axhline(0, color='w', linestyle='--')
        ax2.axhline(0.5, color='w', linestyle='--')
        # plt.autoscale(enable=True)
        ax1.set_xlabel('$\\rho $')
        ax2.set_xlabel('$\\rho $')
        ax1.set_xlim([rho_min, rho_max])
        ax2.set_xlim([rho_min, rho_max])
        ylabels = ('X-DOWN', 'OMP', 'UP', 'HFS', 'X-DOWN')
        y_pos = [-1, -0.5, 0, 0.5, 1]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ylabels)
        ax1.set_title(title1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(ylabels)
        ax2.set_title(title2)
        anim = Player(fig, animate)
        return anim
        #plt.show()


def edge_animation_bar_5(magnitude1, title1, magnitude2, title2, magnitude3, title3, magnitude4, title4, magnitude5, title5):
    cmin = -0.06
    cmax = 0.06
    m1 = filter(magnitude1)
    m2 = filter(magnitude2)
    m3 = filter(magnitude3)
    m4 = filter(magnitude4)
    m5 = filter(magnitude5)
    fig = plt.figure(figsize=(16, 16))
    ax1 = fig.add_subplot(1, 5, 1)
    cax1 = ax1.pcolor(rho, (eta - math.pi) / math.pi, m1[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)
    ax2 = fig.add_subplot(1, 5, 2)
    cax2 = ax2.pcolor(rho, (eta - math.pi) / math.pi, m2[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)
    ax3 = fig.add_subplot(1, 5, 3)
    cax3 = ax3.pcolor(rho, (eta - math.pi) / math.pi, m3[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)
    ax4 = fig.add_subplot(1, 5, 4)
    cax4 = ax4.pcolor(rho, (eta - math.pi) / math.pi, m4[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)
    ax5 = fig.add_subplot(1, 5, 5)
    cax5 = ax5.pcolor(rho, (eta - math.pi) / math.pi, m5[1, :-1, :-1], cmap='jet', vmin=cmin, vmax=cmax)

    def animate(i):
        cax1.set_array(m1[i + 1, :-1, :-1].flatten('C'))
        cax2.set_array(m2[i + 1, :-1, :-1].flatten('C'))
        cax3.set_array(m3[i + 1, :-1, :-1].flatten('C'))
        cax4.set_array(m4[i + 1, :-1, :-1].flatten('C'))
        cax5.set_array(m5[i + 1, :-1, :-1].flatten('C'))

    fig.colorbar(cax1)
    ax1.axvline(x=1, color='k', linestyle='--')
    ax1.axhline(-0.5, color='w', linestyle='--')
    ax1.axhline(0, color='w', linestyle='--')
    ax1.axhline(0.5, color='w', linestyle='--')
    ax2.axvline(x=1, color='k', linestyle='--')
    ax2.axhline(-0.5, color='w', linestyle='--')
    ax2.axhline(0, color='w', linestyle='--')
    ax2.axhline(0.5, color='w', linestyle='--')
    ax3.axvline(x=1, color='k', linestyle='--')
    ax3.axhline(-0.5, color='w', linestyle='--')
    ax3.axhline(0, color='w', linestyle='--')
    ax3.axhline(0.5, color='w', linestyle='--')
    ax4.axvline(x=1, color='k', linestyle='--')
    ax4.axhline(-0.5, color='w', linestyle='--')
    ax4.axhline(0, color='w', linestyle='--')
    ax4.axhline(0.5, color='w', linestyle='--')
    ax5.axvline(x=1, color='k', linestyle='--')
    ax5.axhline(-0.5, color='w', linestyle='--')
    ax5.axhline(0, color='w', linestyle='--')
    ax5.axhline(0.5, color='w', linestyle='--')
    # plt.autoscale(enable=True)
    ax1.set_xlabel('$\\rho $')
    ax2.set_xlabel('$\\rho $')
    ax3.set_xlabel('$\\rho $')
    ax4.set_xlabel('$\\rho $')
    ax5.set_xlabel('$\\rho $')
    ax1.set_xlim([rho_min, rho_max])
    ax2.set_xlim([rho_min, rho_max])
    ax3.set_xlim([rho_min, rho_max])
    ax4.set_xlim([rho_min, rho_max])
    ax5.set_xlim([rho_min, rho_max])
    ylabels = ('X-DOWN', 'OMP', 'UP', 'HFS', 'X-DOWN')
    y_pos = [-1, -0.5, 0, 0.5, 1]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels)
    ax1.set_title(title1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ylabels)
    ax2.set_title(title2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(ylabels)
    ax3.set_title(title3)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(ylabels)
    ax4.set_title(title4)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(ylabels)
    ax5.set_title(title5)
    anim = Player(fig, animate)
    return anim
    # plt.show()


def radial_plot(magnitude):
    m=filter(magnitude)
    plt.figure()
    plt.plot(rho, m[:][0])
    plt.plot(rho, m[:][480])
    plt.plot(rho, m[:][480 * 2])
    plt.plot(rho, m[:][480 * 3])
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(['DOWN', 'LFS', 'UP', 'HFS'])
    plt.autoscale(enable=True, axis='y')
    plt.xlim([rho_min, rho_max])

def time_average(magnitude):
    return np.mean(magnitude, axis=0)

rho_min = 0.6
rho_max = 1.1


dt_Omega_E=ds['v_Omega_E_2dX'][:]
dt_Omega_E_gf=ds['v_Omega_E_gf_2dX'][:]
dt_Omega_E_TK=ds['v_Omega_E_cta2d'][:]
dt_Omega_E_TK_fluct=ds['v_Omega_E_fluc2d'][:]
dt_Omega_E_TK=ds['v_Omega_E_gf_cta2d'][:]
dt_Omega_E_TK_fluct=ds['v_Omega_E_gf_fluc2d'][:]
dt_Omega_E_gf=ds['v_Omega_E_gf_2dX'][:]
dt_Omega_E_fsa=ds['v_Omega_E_fsa'][:]
dt_Omega_E_gf_fsa=ds['v_Omega_E_gf_fsa'][:]
dt_Omega_D=ds['v_Omega_D_2dX'][:]
dt_Omega_D_gf=ds['v_Omega_D_gf_2dX'][:]
dt_Omega_D_fsa=ds['v_Omega_D_fsa'][:]
dt_Omega_D_gf_fsa=ds['v_Omega_D_gf_fsa'][:]
dt_Omega_D_TK=ds['v_Omega_D_cta2d'][:]
dt_Omega_D_TK_fluct=ds['v_Omega_D_fluc2d'][:]

for i in range(1, t.size):
    dt_Omega_E[:][i-1] = ds['v_Omega_E_2dX'][:][i] - ds['v_Omega_E_2dX'][:][i - 1]
    dt_Omega_D[:][i-1] = ds['v_Omega_D_2dX'][:][i] - ds['v_Omega_D_2dX'][:][i - 1]
    dt_Omega_E_gf[:][i - 1] = ds['v_Omega_E_gf_2dX'][:][i] - ds['v_Omega_E_gf_2dX'][:][i - 1]
    dt_Omega_D_gf[:][i - 1] = ds['v_Omega_D_gf_2dX'][:][i] - ds['v_Omega_D_gf_2dX'][:][i - 1]
    dt_Omega_E_fsa[:][i - 1] = ds['v_Omega_E_fsa'][:][i] - ds['v_Omega_E_fsa'][:][i - 1]
    dt_Omega_D_fsa[:][i - 1] = ds['v_Omega_D_fsa'][:][i] - ds['v_Omega_D_fsa'][:][i - 1]
    dt_Omega_E_gf_fsa[:][i - 1] = ds['v_Omega_E_gf_fsa'][:][i] - ds['v_Omega_E_gf_fsa'][:][i - 1]
    dt_Omega_D_gf_fsa[:][i - 1] = ds['v_Omega_D_gf_fsa'][:][i] - ds['v_Omega_D_gf_fsa'][:][i - 1]
    dt_Omega_E_TK[:][i - 1] = ds['v_Omega_E_cta2d'][:][i] - ds['v_Omega_E_cta2d'][:][i - 1]
    dt_Omega_D_TK[:][i - 1] = ds['v_Omega_D_cta2d'][:][i] - ds['v_Omega_D_cta2d'][:][i - 1]
    #dt_Omega_E_TK_gf[:][i - 1] = ds['v_Omega_E_gf_cta2d'][:][i] - ds['v_Omega_E_gf_cta2d'][:][i - 1]
    #dt_Omega_D_TK_gf[:][i - 1] = ds['v_Omega_D_gf_cta2d'][:][i] - ds['v_Omega_D_gf_cta2d'][:][i - 1]
    dt_Omega_E_TK_fluct[:][i - 1] = ds['v_Omega_E_fluc2d'][:][i] - ds['v_Omega_E_fluc2d'][:][i - 1]
    dt_Omega_D_TK_fluct[:][i - 1] = ds['v_Omega_D_fluc2d'][:][i] - ds['v_Omega_D_fluc2d'][:][i - 1]

dt_Omega_E_ta=time_average(dt_Omega_E)
dt_Omega_D_ta=time_average(dt_Omega_D)
dt_Omega_E_gf_ta=time_average(dt_Omega_E_gf)
dt_Omega_D_gf_ta=time_average(dt_Omega_D_gf)
dt_Omega = dt_Omega_E + dt_Omega_D
dt_Omega_fsa = dt_Omega_E_fsa + dt_Omega_D_fsa
dt_Omega_gf = dt_Omega_E_gf + dt_Omega_D_gf
dt_Omega_gf_fsa = dt_Omega_E_gf_fsa + dt_Omega_D_gf_fsa
#dt_Omega_TK=dt_Omega_E_TK+dt_Omega_D_TK
#dt_Omega_TK_gf=dt_Omega_E_TK_gf+dt_Omega_D_TK_gf

#edge_animation_bar_2(dt_Omega, 'dt_Omega', dt_Omega_gf, 'dt_Omega_gf')


electric_adv = ds['v_adv_E_tt_2dX'][:]
dielectric_adv = ds['v_adv_D_tt_2dX'][:]
electric_adv_TK = ds['v_adv_E_tt_cta2d'][:]
dielectric_adv_TK = ds['v_adv_D_tt_cta2d'][:]
electric_adv_TK_gf = ds['v_adv_E_gf_tt_cta2d'][:]
dielectric_adv_TK_gf = ds['v_adv_D_gf_tt_cta2d'][:]
electric_adv_gf = ds['v_adv_E_gf_tt_2dX'][:]
dielectric_adv_gf = ds['v_adv_D_gf_tt_2dX'][:]
electric_adv_fsa = ds['v_adv_E_tt_fsa'][:]
dielectric_adv_fsa = ds['v_adv_D_tt_fsa'][:]
electric_adv_gf_fsa = ds['v_adv_E_gf_tt_fsa'][:]
dielectric_adv_gf_fsa = ds['v_adv_D_gf_tt_fsa'][:]

e_adv_ta=time_average(electric_adv)
d_adv_ta=time_average(dielectric_adv)
e_adv_gf_ta=time_average(electric_adv_gf)
d_adv_gf_ta=time_average(dielectric_adv_gf)
extra_gf = ds['v_EXTRA_GF_tt_2dX'][:]
extra_gf_fsa = ds['v_EXTRA_GF_tt_fsa'][:]
extra_gf_ta=np.mean(extra_gf, axis=0)

#edge_animation_bar(extra_gf, 'EXTRA GF TERM')

advection = electric_adv + dielectric_adv
advection_TK = electric_adv_TK + dielectric_adv_TK
advection_gf = electric_adv_gf + dielectric_adv_gf
advection_fsa = electric_adv_fsa + dielectric_adv_fsa
advection_gf_fsa = electric_adv_gf_fsa + dielectric_adv_gf_fsa
LHS = dt_Omega + advection
#LHS_TK = dt_Omega_TK + advection_TK
#LHS_TK_gf = dt_Omega_TK_gf + advection_TK_gf
LHS_gf = dt_Omega_gf + advection_gf + extra_gf
LHS_fsa = dt_Omega_fsa + advection_fsa
LHS_gf_fsa = dt_Omega_gf_fsa + advection_gf_fsa + extra_gf_fsa

#LHS_TK = dt_Omega_TK + advection_TK
LHS_fsa_ta=np.mean(LHS_fsa, axis=0)
LHS_gf_fsa_ta=np.mean(LHS_gf_fsa, axis=0)
LHS_gf_ta=np.mean(LHS_gf, axis=0)
LHS_ta=np.mean(LHS, axis=0)

#edge_animation_bar_2(LHS, 'LHS', LHS_gf, 'LHS_gf')


J_par = ds['v_J_par_tt_2dX'][:]
J_par_TK = ds['v_J_par_tt_cta2d'][:]
J_par_fsa = ds['v_J_par_tt_fsa'][:]

J_par_fsa_ta=np.mean(J_par_fsa, axis=0)

fluct_1 = ds['v_J_bperp_tt_2dX'][:]
fluct_2 = ds['v_J_mag_tt_2dX'][:]
fluct_3 = ds['v_M_em_tt_2dX'][:]
fluct_1_TK = ds['v_J_bperp_tt_cta2d'][:]
fluct_2_TK = ds['v_J_mag_tt_cta2d'][:]
fluct_3_TK = ds['v_M_em_tt_cta2d'][:]
fluct_1_gf = ds['v_J_bperp_gf_tt_2dX'][:]
fluct_2_gf = ds['v_J_mag_gf_tt_2dX'][:]
fluct_3_gf = ds['v_M_em_gf_tt_2dX'][:]
fluct_1_fsa = ds['v_J_bperp_tt_fsa'][:]
fluct_2_fsa = ds['v_J_mag_tt_fsa'][:]
fluct_3_fsa = ds['v_M_em_tt_fsa'][:]
fluct_1_gf_fsa = ds['v_J_bperp_gf_tt_fsa'][:]
fluct_2_gf_fsa = ds['v_J_mag_gf_tt_fsa'][:]
fluct_3_gf_fsa = ds['v_M_em_gf_tt_fsa'][:]
J_b_perp = -fluct_1 - fluct_2 + fluct_3
J_b_perp_TK = -fluct_1_TK - fluct_2_TK + fluct_3_TK
J_b_perp_gf = -fluct_1_gf - fluct_2_gf + fluct_3_gf
J_b_perp_fsa = -fluct_1_fsa - fluct_2_fsa + fluct_3_fsa
J_b_perp_gf_fsa = -fluct_1_gf_fsa - fluct_2_gf_fsa + fluct_3_gf_fsa

curv_1 = ds['v_J_D_tt_2dX'][:]
curv_2 = ds['v_J_JAK_tt_2dX'][:]
curv_3 = ds['v_J_NUK_tt_2dX'][:]
curv_1_TK = ds['v_J_D_tt_cta2d'][:]
curv_2_TK = ds['v_J_JAK_tt_cta2d'][:]
curv_3_TK = ds['v_J_NUK_tt_cta2d'][:]
curv_1_gf = ds['v_J_D_gf_tt_2dX'][:]
curv_2_gf = ds['v_J_JAK_gf_tt_2dX'][:]
curv_3_gf = ds['v_J_NUK_gf_tt_2dX'][:]
curv_1_fsa = ds['v_J_D_tt_fsa'][:]
curv_2_fsa = ds['v_J_JAK_tt_fsa'][:]
curv_3_fsa = ds['v_J_NUK_tt_fsa'][:]
curv_1_gf_fsa = ds['v_J_D_gf_tt_fsa'][:]
curv_2_gf_fsa = ds['v_J_JAK_gf_tt_fsa'][:]
curv_3_gf_fsa = ds['v_J_NUK_gf_tt_fsa'][:]
J_curv = curv_1 + curv_2 + curv_3
J_curv_TK = curv_1_TK + curv_2_TK + curv_3_TK
J_curv_gf = curv_1_gf + curv_2_gf + curv_3_gf
J_curv_fsa = curv_1_fsa + curv_2_fsa + curv_3_fsa
J_curv_gf_fsa = curv_1_gf_fsa + curv_2_gf_fsa + curv_3_gf_fsa

J_curv_fsa_ta=np.mean(J_curv_fsa, axis=0)
J_curv_gf_fsa_ta=np.mean(J_curv_gf_fsa, axis=0)


RHS=J_par+J_b_perp+J_curv
RHS_TK=J_par_TK+J_b_perp_TK+J_curv_TK
RHS_gf=J_par+J_b_perp_gf+J_curv_gf
RHS_fsa=J_par_fsa+J_b_perp_fsa+J_curv_fsa
RHS_gf_fsa=J_par_fsa+J_b_perp_gf_fsa+J_curv_gf_fsa

RHS_gf_ta=np.mean(RHS_gf, axis=0)
RHS_ta=np.mean(RHS, axis=0)
RHS_fsa_ta=np.mean(RHS_fsa, axis=0)
RHS_gf_fsa_ta=np.mean(RHS_gf_fsa, axis=0)

#edge_animation_bar_5(dt_Omega, r'$\partial_t\Omega_E$', advection, r'$-\nabla\cdot\nabla\cdot(\omega u_E)$', J_par, r'$\nabla\cdot(j_\parallel \hat{b})$', J_curv,r'$\nabla\cdot j_{curv}$', J_b_perp, r'$\nabla\cdot j_{b_\perp}$' )




'''
E_r = ds['RFB_E_r_tt_2dX'][:]
# edge_plot(E_r)
# plt.title(r'$E_r$')
dP_dr = ds['RFB_GradPi_tt_2dX'][:]
# edge_plot(dP_dr)
# plt.title(r'$\partial P_i/\partial r$')
'''


'''
fig = plt.figure(figsize=(16, 16))
fig.suptitle('RHS Conservation of currents EQ')
ax1 = fig.add_subplot(1, 5, 1)
p1 = edge_plot(vort_elec, r'$\partial_t\Omega_E$', ax1)
fig.colorbar(p1)
ax2 = fig.add_subplot(1, 5, 2)
p2 = edge_plot(vort_dielec, r'$\partial_t\Omega_D$', ax2)
fig.colorbar(p2)
ax3 = fig.add_subplot(1, 5, 3)
p3 = edge_plot(electric_adv, r'$\nabla \cdot \nabla \cdot (\omega_E u_E)$', ax3)
fig.colorbar(p3)
ax4 = fig.add_subplot(1, 5, 4)
p4 = edge_plot(dielectric_adv, r'$\nabla \cdot \nabla \cdot (\omega_D u_E)$', ax4)
fig.colorbar(p4)
ax5 = fig.add_subplot(1, 5, 5)
p5 = edge_plot(LHS, 'LHS', ax5)
fig.colorbar(p5)
#fig.tight_layout()
fig.show()
#E_r = ds['RFB_E_r_tt_2dX'][:]
# edge_plot(E_r)
# plt.title(r'$E_r$')
#dP_dr = ds['RFB_GradPi_tt_2dX'][:]
# edge_plot(dP_dr)
# plt.title(r'$\partial P_i/\partial r$')
RHS=J_par+J_b_perp+J_curv
fig = plt.figure(figsize=(16, 16))
fig.suptitle('RHS Conservation of currents EQ')
ax1 = fig.add_subplot(1, 4, 1)
p1 = edge_plot(J_par, 'J_par', ax1)
fig.colorbar(p1)
ax2 = fig.add_subplot(1, 4, 2)
p2 = edge_plot(J_b_perp, 'J_b_perp', ax2)
fig.colorbar(p2)
ax3 = fig.add_subplot(1, 4, 3)
p3 = edge_plot(J_curv, 'J_curv', ax3)
fig.colorbar(p3)
ax4 = fig.add_subplot(1, 4, 4)
p4 = edge_plot(J_b_perp, 'RHS', ax4)
fig.colorbar(p4)
#fig.tight_layout()
fig.show()
fig = plt.figure(figsize=(8, 8))
fig.suptitle('Conservation of currents EQ')
ax1 = fig.add_subplot(1, 2, 1)
p1 = edge_plot(LHS, 'LHS', ax1)
fig.colorbar(p1)
ax2 = fig.add_subplot(1, 2, 2)
p2 = edge_plot(RHS, 'rhs', ax2)
fig.colorbar(p2)
#fig.tight_layout()
fig.show()
'''

'''
u_E_tor=ds['u_E_tor_tt_2dX'][:] 
u_E=ds['u_E_tt_2dX'][:] 
u_E_pol=np.sqrt(ds['u_E_tt_2dX'][:][5]**2-ds['u_E_tor_tt_2dX'][:] **2)
edge_plot(u_E)
plt.title('u_E')
edge_plot(u_E_tor)
plt.title('u_E_tor')
edge_plot(u_E_pol)
plt.title('u_E_pol')
   
    elec_source= C*ds['v_S_E_tt_2dX'][:]
    dielec_source=C*tau*ds['v_S_D_tt_2dX'][:]
    Sources=elec_source+dielec_source
    
    edge_plot(Sources)
    plt.title(r'$\Omega_S$')
 
    
    plt.pcolor(rho[(rho>rho_min) & (rho<rho_max)], (eta-math.pi)/math.pi, curv_1[0][:, (rho>rho_min) & (rho<rho_max)], cmap='jet', shading='auto' )
    plt.axvline(x=1, color='k', linestyle='--')
    plt.axhline(-0.5, color='w', linestyle='--')
    plt.axhline(0, color='w', linestyle='--')
    plt.axhline(0.5, color='w', linestyle='--')
    plt.autoscale(enable=True)
    plt.colorbar()
    plt.xlabel('$\\rho $')
    #plt.ylabel('$\\theta/\\pi$')
    #plt.clim(-1e10, 1e10)
    ylabels=('X-point', 'HFS', 'UP', 'LFS', 'X-point')
    y_pos=[-1, -0.5, 0, 0.5, 1]
    plt.yticks(y_pos, ylabels)
    #plt.xlim(-0.5, 0.5)
'''
'''
#LIST OF CONCLUSIONS
1. VORTICITY RADIAL COMPONENT IS THE MAIN ONE (Except for little fluctuations)
2. Same for ADVECTION terms
3. The main components of the advections are small compared with the complete, although they have structure in the order of 0.03
CONCLUSIONS FROM ADVECTION
1. w_E*nabla u_E is small compared with W_E u_E, and therefore, it's divergence too, so the alt electric term makes sense to be so small
2. the radial component of W_E u_E is small, which makes sense, as u_E will be bigger in the perpendicular direction, not the radial.
3. In the diamagnetic case, the w_D*nabla u_E is smaller than the other component, but larger than w_E*nabla u_E (order 10^-3 compared with 10^-7)
4. W_D u_E is smaller in the radial direction than the total, but still relevant, with a strong poloidal distribution
5. the dielectric advective main term seems to be the radial direction of the radial componend of adv_WD_UE_r, so we can trust it
6. Mem nabla b is extremely small (order 10^-9)
*LIttle difference between grad B and curv term

