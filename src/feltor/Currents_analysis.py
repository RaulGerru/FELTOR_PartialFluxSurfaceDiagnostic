# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:41:10 2021
ANALYSIS PROGRAM FOR VORTICITY EQUATION IN FELTOR: 
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

# fn="Final_Test_1X_simple_diagRaul_FINAL2.nc"
fn = "/home/raulgerru/Desktop/PhD files/Research/FELTOR/SIMULATIONS/Diag_test_files/conservation_test_diag.nc"
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
# tau=tau=inputfile_json['physical']['tau']
tau = 7
Ti = T0 * tau

Omega_0 = e * B_0 / (m_i * m_H)
C = e * n0 * Omega_0

rho = ds['rho'][:]
eta = ds['eta'][:]  # Poloidal direction (from 0 to 2pi)
t = ds['time'][:]
t_def = 5
time = 1e3 * ds['time'][:] / Omega_0
density = ds['electrons_2dX'][:][t_def]


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

def filter(data):
    fft = fftpack.fft2(data)
    fft[26:1894] = 0
    data = fftpack.ifft2(fft).real
    return data
'''
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=t.size, pos=(0.125, 0.08), **kwargs):
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
'''
def edge_plot(magnitude, title, axes=None):
    if axes is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1 = axes
    #cmin = -0.03
    #cmax = 0.05
    p = plt.pcolor(rho[(rho > rho_min) & (rho < rho_max)], (eta - math.pi) / math.pi,
                   filter(magnitude[:, (rho > rho_min) & (rho < rho_max)]), cmap='jet')#,  vmin=cmin, vmax=cmax)#, shading='gouraud')
    ax1.axvline(x=1, color='k', linestyle='--')
    ax1.axhline(-0.5, color='w', linestyle='--')
    ax1.axhline(0, color='w', linestyle='--')
    ax1.axhline(0.5, color='w', linestyle='--')
    ax1.autoscale(enable=True)
    ax1.set_xlabel('$\\rho $')
    ylabels = ('DOWN', 'LFS', 'UP', 'HFS', 'DOWN')
    y_pos = [-1, -0.5, 0, 0.5, 1]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels)
    ax1.set_title(title)
    return p

def edge_plot_2(magnitude, title, axes=None):
    if axes is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1 = axes
    #cmin = -0.03
    #cmax = 0.05
    p = plt.pcolor(rho[(rho > rho_min) & (rho < rho_max)], (eta - math.pi) / math.pi,
                   filter(magnitude[:, (rho > rho_min) & (rho < rho_max)]), cmap='jet')#,  vmin=cmin, vmax=cmax)#, shading='gouraud')
    ax1.axvline(x=1, color='k', linestyle='--')
    ax1.axhline(-0.5, color='w', linestyle='--')
    ax1.axhline(0, color='w', linestyle='--')
    ax1.axhline(0.5, color='w', linestyle='--')
    ax1.autoscale(enable=True)
    ax1.set_xlabel('$\\rho $')
    ylabels = ('DOWN', 'LFS', 'UP', 'HFS', 'DOWN')
    y_pos = [-1, -0.5, 0, 0.5, 1]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels)
    ax1.set_title(title)
    #fig.colorbar(p)
    return p


def radial_plot(magnitude):
    plt.figure()
    plt.plot(rho, magnitude[:][0])
    plt.plot(rho, magnitude[:][480])
    plt.plot(rho, magnitude[:][480 * 2])
    plt.plot(rho, magnitude[:][480 * 3])
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(['DOWN', 'LFS', 'UP', 'HFS'])
    plt.autoscale(enable=True, axis='y')
    plt.xlim([rho_min, rho_max])


rho_min = 0.6
rho_max = 1.1


vort_elec = ds['v_Omega_E_2dX'][:][t_def] - ds['v_Omega_E_2dX'][:][t_def - 1]
vort_dielec = ds['v_Omega_D_2dX'][:][t_def] - ds['v_Omega_D_2dX'][:][t_def - 1]
dt_Omega = vort_elec + vort_dielec


electric_adv = ds['v_adv_E_tt_2dX'][:][t_def]
#electric_adv_alt = ds['v_adv_E_2dX'][:][t_def]
dielectric_adv = ds['v_adv_D_tt_2dX'][:][t_def]
#dielectric_adv_alt = ds['v_adv_D_2dX'][:][t_def]
advection = electric_adv + dielectric_adv
#advection_alt = electric_adv_alt + dielectric_adv_alt
LHS = dt_Omega + advection
#LHS_alt=dt_Omega_alt + advection_alt

#fig = plt.figure(figsize=(16, 16))
#fig.suptitle('LHS Conservation of currents EQ')
#ax5 = fig.add_subplot(1, 2, 1)
#p5 = edge_plot(LHS, 'LHS', ax5)
#fig.colorbar(p5)
#ax1 = fig.add_subplot(1, 2, 2)
#p1 = edge_plot(LHS_alt, 'LHS alt', ax1)
#fig.colorbar(p5)


fig = plt.figure(figsize=(16, 16))
fig.suptitle('LHS Conservation of currents EQ')
ax5 = fig.add_subplot(1, 5, 5)
p5 = edge_plot(LHS, 'LHS', ax5)
fig.colorbar(p5)
ax1 = fig.add_subplot(1, 5, 1)
p1 = edge_plot(vort_elec, r'$\partial_t\Omega_E$', ax1)
fig.colorbar(p5)
ax2 = fig.add_subplot(1, 5, 2)
p2 = edge_plot(vort_dielec, r'$\partial_t\Omega_D$', ax2)
fig.colorbar(p5)
ax3 = fig.add_subplot(1, 5, 3)
p3 = edge_plot(electric_adv, r'$\nabla \cdot \nabla \cdot (\omega_E u_E)$', ax3)
fig.colorbar(p5)
ax4 = fig.add_subplot(1, 5, 4)
p4 = edge_plot(dielectric_adv, r'$\nabla \cdot \nabla \cdot (\omega_D u_E)$', ax4)
fig.colorbar(p5)
fig.show()

J_par = ds['v_J_par_tt_2dX'][:][t_def]
#J_par_alt = ds['v_J_par_2dX'][:][t_def]

fluct_1 = ds['v_J_bperp_tt_2dX'][:][t_def]
fluct_2 = ds['v_J_mag_tt_2dX'][:][t_def]
fluct_3 = ds['v_M_em_tt_2dX'][:][t_def]
J_b_perp = -fluct_1 - fluct_2 + fluct_3



curv_1 = ds['v_J_D_tt_2dX'][:][t_def]
curv_2 = ds['v_J_JAK_tt_2dX'][:][t_def]
curv_3 = ds['v_J_NUK_tt_2dX'][:][t_def]

J_curv = curv_1 + curv_2 + curv_3


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
p4 = edge_plot(RHS, 'RHS', ax4)
fig.colorbar(p4)
#fig.tight_layout()
fig.show()

diffusion = ds['v_L_i_perp_tt_2dX'][:][t_def]


fig = plt.figure(figsize=(16, 16))
fig.suptitle('Conservation of currents EQ')
ax1 = fig.add_subplot(1, 4, 1)
p1 = edge_plot(LHS, 'LHS', ax1)
fig.colorbar(p1)
ax2 = fig.add_subplot(1, 4, 2)
p2 = edge_plot(RHS, 'rhs', ax2)
fig.colorbar(p2)
ax3 = fig.add_subplot(1, 4, 3)
p3 = edge_plot(LHS-RHS, 'lhs-rhs', ax3)
fig.colorbar(p3)
ax4 = fig.add_subplot(1, 4, 4)
p4 = edge_plot(diffusion, 'diffusion', ax4)
fig.colorbar(p4)
#fig.tight_layout()
fig.show()

E_r = ds['RFB_E_r_tt_2dX'][:][t_def]

dP_dr = ds['RFB_GradPi_tt_2dX'][:][t_def]




'''
u_E_tor=ds['u_E_tor_tt_2dX'][:][t_def]
u_E=ds['u_E_tt_2dX'][:][t_def]
u_E_pol=np.sqrt(ds['u_E_tt_2dX'][:][5]**2-ds['u_E_tor_tt_2dX'][:][t_def]**2)
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
'''


