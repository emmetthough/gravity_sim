import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np

from scipy.interpolate import interp2d, RectBivariateSpline
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import time
import os
# from basic_units import radians, degrees, cos
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from joblib import Parallel, delayed
import itertools

class attractor_profile:
    
    def __init__(self, R, z_size=1, dr=1, dz=1, dphi=1, r_range=1.2):
        
        # give all params in microns
        
        self.is_built = False
        self.data = {}
        self.dphi_i = dphi # in um
        self.R = R # in um
        self.M = 1e3*z_size*1e-6*np.pi*(R*1e-6)**2 # test for uniform mass
        
        # create partitions
        
        self.n_r = round((R - 0.5*dr) / dr)
        self.dr_dyn = (R - 0.5*dr) / self.n_r
        
        self.n_z = round(z_size / dz)
        self.dz_dyn = z_size / self.n_z
        
        # Center points of radial partitions, in m
        rr = np.linspace(self.dr_dyn, R-(self.dr_dyn/2), self.n_r)
        rr_excess = np.arange(R, R*r_range, self.dr_dyn)+self.dr_dyn/2
        self.rr = np.concatenate(([0], rr, rr_excess))*1e-6
        
        # Center points of z partitons, in m
        self.zz = np.linspace(-z_size/2 + self.dz_dyn/2, z_size/2 - self.dz_dyn/2, self.n_z)*1e-6
        
    def build(self, density_profile):
        
        # density_profile is function that returns density when passed (r,phi,z)
        # working in kms
        
        self.density_profile = density_profile
        z = 0. # considering cylindrical symmetry for now, can add wrapper for loop in z later
        self.sum_dm = 0.
        
        # dr, dz in um: convert
        dr = self.dr_dyn*1e-6
        dz = self.dz_dyn*1e-6
        
        for r in tqdm_notebook(self.rr, desc='Building Attractor'):
            
            radial_partition = {}
            
            # create angular partition, r in m
            if r != 0.:
                
                n_phi = round(2*np.pi*(r*1e6) / self.dphi_i) # dphi_i given in microns, so convert r back to microns
                dphi_dyn = 2*np.pi / n_phi # in rads
                pp = np.linspace(dphi_dyn/2, 2*np.pi - dphi_dyn/2, n_phi)
                            
                data = np.zeros((self.zz.size, n_phi))
                dV = r*dr*dphi_dyn*dz
                
                for i,z in enumerate(self.zz):
                    for j,phi in enumerate(pp):
                        rho = density_profile(r, phi, z) # in kg/m^3
                        dm = rho*dV # kg
                        data[i,j] = dm
                        self.sum_dm += dm

                radial_partition = {'params': (self.dr_dyn, dphi_dyn, self.dz_dyn, pp),
                                       'data': data}

            else: # r == 0
                
                data = np.zeros(self.zz.size)
                dV = np.pi*(dr/2)**2 * dz # m^3
                
                for i,z in enumerate(self.zz):
                    rho = density_profile(0.,0.,z) # kg/m^3
                    dm = rho*dV  # kg
                    data[i] = dm
                    self.sum_dm += dm
                    
                radial_partition = {'params': (self.dr_dyn/2, 2*np.pi, self.dz_dyn, np.array([0.])), 
                                 'data': data} # phi value for center? None?

            self.data[r] = radial_partition
        
        self.is_built = True
        # only for uniform test
        # self.sum_dm *= len(self.zz)
        
    def plot_xy(self, upsample=10, show_data=False):
        
        # convert angular data to cartesian and display
        # converting back to um for ease
        
        rr = np.array([0.])
        pp = np.array([0.])
        mm = np.array([self.data[0.]['data'][0]])
        
        xxi = np.array([0.])
        yyi = np.array([0.])
        
        # create full coordinate list of r, phi, rho
        for r, partition in self.data.items():
            data_arr = partition['data']
            pp_i = partition['params'][3]
            if pp_i.size != 1:
                dms = data_arr[0,:]

                rr = np.concatenate((rr, np.full(dms.size, r*1e6)))
                pp = np.concatenate((pp, pp_i))
                mm = np.concatenate((mm, dms))

                xxi = np.concatenate((xxi, r*1e6*np.cos(pp_i)))
                yyi = np.concatenate((yyi, r*1e6*np.sin(pp_i)))
            
                    
        coord_data = (rr, pp, mm)
                        
        dx = self.dr_dyn / upsample
        xx = np.arange(-self.rr[-1]*1e6, self.rr[-1]*1e6+dx, dx)
        yy = xx
        rho_cart = np.zeros((xx.size, yy.size))
        
        for i in tqdm_notebook(np.arange(xx.size), desc='Building Attractor Image'):
            
            x = xx[i]
            for j,y in enumerate(yy):
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                if phi < 0:
                    phi += 2*np.pi

                rho_cart[j,i] =  self.density_profile(r*1e-6,phi,0.) #check index order
                
        fig, ax = plt.subplots(1, figsize=(12,12))      
        ax.set_ylim(-100,100)
        ax.set_xlim(7900, 8100)
        ax.contourf(xx,yy,rho_cart)
        # ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        
        
    def newtonian(self, rb_vec, Rb=5e-6, rhob=1850.0, debug=False):
        
        # rb_vec given in um, tuple
        
        G = 6.67e-11
        Fg = np.array([0.,0.,0.])
        rb, pb, zb = rb_vec
        rb = rb*1e-6
        zb = zb*1e-6
        
        i = -1
        
        for r, partition in self.data.items():       
            
            dm_arr = partition['data'].T # transpose here for shape, fix in build attractor?
            pp = partition['params'][3]
                     
            # create coordinate arrays to compute force vector at
            psep, zsep = np.meshgrid(pp - pb, self.zz - zb, indexing='ij')
            rsep = rb - r
                       
            # separation between attractor mesh and bead
            sep = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep) + zsep**2)
            
            # separation in only r,phi
            sep_rp = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep))
                    
            # get full vector force at every point in meshgrid (i.e. every r,phi,z in partition)
            full_vec_force = -1.0 * (4.*G*dm_arr*rhob*np.pi*Rb**3)/(3.*sep**2)
            
            # add 1% white noise
            full_vec_force += np.random.randn()*full_vec_force/100
            
            # get projections onto each unit vector and sum the force at all points on the partition
            Fg[0] += np.sum(full_vec_force * (rb-r*np.cos(psep))/sep_rp)
            Fg[1] += np.sum(full_vec_force * (r*np.sin(psep))/sep_rp)
            Fg[2] += np.sum(full_vec_force * (zsep/sep))   
            
            if debug:
                print('r: ', r)
                print('pp: ', pp*180/np.pi)
                print('dm_arr: ', dm_arr)
                print('rsep: ', rsep)
                print('psep: ', psep*180/np.pi)
                print('zsep: ', zsep)
                print('sep: ', sep)
                print('full_vec_force: ', full_vec_force)
                print()
        
        return Fg
    
    
    def yukawa(self, rb_vec, a=1., l=100e-6, Rb=5e-6, rhob=1850.0):
        
        # given bead position rb_vec, compute yukawa force
        
        G = 6.67e-11
        Fg = np.array([0.,0.,0.])
        
        rb, pb, zb = rb_vec
        rb = rb*1e-6
        zb = zb*1e-6
        
        func = np.exp(-2. * Rb/l) * (1. + Rb/l) + Rb/l - 1.
        
        for r, partition in self.data.items():
            
            dm_arr = partition['data'].T # transpose here for shape, fix in build attractor?
            pp = partition['params'][3]
            
            # create coordinate arrays to compute force vector at
            psep, zsep = np.meshgrid(pp - pb, self.zz - zb, indexing='ij')
            rsep = rb - r
                       
            # separation between attractor mesh and SURFACE of bead
            sep = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep) + zsep**2) - Rb
            
            # separation in only r,phi; to center for projections
            sep_rp = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep))
            
            # yukawa terms (why?)
            # where is alpha??
            prefac = -((2.*G*dm_arr*rhob*np.pi) / (3.*(sep+Rb)**2))
            yukterm = 3.*l**2 * (sep+Rb+l) * func * np.exp(-sep/l)
            
            # get full vector force at every point in meshgrid (i.e. every r,phi,z in partition)
            full_vec_force = prefac * yukterm
            
            # add 1% white noise
            full_vec_force += np.random.randn()*full_vec_force/100
            
            # get projections onto each unit vector and sum the force at all points on the partition
            Fg[0] += np.sum(full_vec_force * (rb-r*np.cos(psep))/sep_rp)
            Fg[1] += np.sum(full_vec_force * (r*np.sin(psep))/sep_rp)
            Fg[2] += np.sum(full_vec_force * (zsep/(sep+Rb)))               
        
        return Fg
    
    
    def force_curve(self, sep, w, z=0., tint=10, fsamp=10e3, time_readout=False):
        
        # should implement some sampling condition with dphi here
        dphi = 2*np.pi*w/fsamp
        phis = np.arange(0, 2*np.pi+dphi, dphi)
        
        Fg_newtonian = [[],[],[]]
        Fg_yukawa = [[],[],[]]
        
        start = time.time()
        
        for phi in tqdm_notebook(phis, desc='Building Full Force Curve'):
            Fg_n = self.newtonian((self.R+sep, phi, z))
            Fg_y = self.yukawa((self.R+sep, phi, z))
            
            for i in range(3):
                Fg_newtonian[i].append(Fg_n[i])
                Fg_yukawa[i].append(Fg_y[i])
                
        end = time.time()
        
        comptime = end-start
        N = phis.size*len(self.data.keys())
        
        if time_readout:
            print('{} computations took {:.3f}s'.format(N, comptime))
                
        Fg_newtonian = np.array(Fg_newtonian)
        Fg_yukawa = np.array(Fg_yukawa)
        
        if tint > 1:
            for i in range(tint-1):
                Fg_newtonian = np.hstack((Fg_newtonian, Fg_newtonian[:,1:]))
                Fg_yukawa = np.hstack((Fg_yukawa, Fg_yukawa[:,1:]))
        
        return Fg_newtonian, Fg_yukawa, (N, comptime)

    
def constant(r,phi,z,R=1000):
    if r <= (R*1e-6): # in m
        return 1e3
    else:
        return 0.
    
R = 1000
uniform = attractor_profile(R, 10)
uniform.build(constant)

seps = np.linspace(5, 500, 100)
    
def phi_vars(sep):
    Fg_phi = []
    for phi in np.linspace(0, 2*np.pi, 25):
        Fg_phi.append(uniform.newtonian((R+sep, phi, 0.)))
    fname = 'gravity_sim/phi_vars/sep_{}_um.npy'.format(sep)
    np.save(fname, Fg_phi)
    
Parallel(n_jobs=20)(delayed(phi_vars)(sep) for sep in tqdm(seps))