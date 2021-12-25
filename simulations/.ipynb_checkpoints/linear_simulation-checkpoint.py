import numpy as np

from tqdm import tqdm
import time
import os
# from basic_units import radians, degrees, cos
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from joblib import Parallel, delayed
import itertools

R = 8000

rho_gold = 19300.
rho_silicon = 2532.59

n_gold_fingers = 9
n_silicon_fingers = n_gold_fingers-1
n_outer = 2

dphi_finger = 0.003125
phi_bound = 9.5*dphi_finger

len_finger = 80*1e-6

center = (dphi_finger/2, 2*np.pi-dphi_finger/2)
boundaries = [dphi_finger*(1/2 + n) for n in np.arange(10)]

gold_lower = [boundaries[i] for i in np.arange(1,9,2)]
gold_upper = [boundaries[i] for i in np.arange(2,9,2)]

silicon_lower = [boundaries[i] for i in np.arange(0,10,2)]
silicon_upper = [boundaries[i] for i in np.arange(1,10,2)]

def in_gold(phi):
    for lower, upper in zip(gold_lower, gold_upper):
        if phi < upper and phi > lower:
            return True
    return False

def in_silicon(phi):
    for lower, upper in zip(silicon_lower, silicon_upper):
        if phi < upper and phi > lower:
            return True
    return False

def in_phi_bounds(phi):
    if phi < phi_bound or phi > (2*np.pi - phi_bound):
        return True
    else:
        return False

def linear_fingers(r,phi,z, R=8000):
    
    R *= 1e-6
    
    if not in_phi_bounds(phi):
        return 0.
    
    if r < (R-len_finger) or r > R:
        return 0.
    
    if in_phi_bounds(phi):
        
        if r >= (R-2e-6): # bridge
            return rho_silicon
    
        if phi < center[0] or phi > center[1]:
            return rho_gold

        if in_gold(phi) or in_gold(2*np.pi - phi):
            return rho_gold

        if in_silicon(phi) or in_silicon(2*np.pi - phi):
            return rho_silicon
    
    return 0.

class attractor_profile:
    
    def __init__(self, R=8000, z_size=9, dr=1, dz=1, dphi=1):
        
        # give all params in microns
        
        self.is_built = False
        self.data = {}
        self.dphi_i = dphi # in um
        self.R = R # in um
        self.finger_length = 80 # in um
        self.length = 100
        
        # create partitions
        
        self.n_r = round(self.length / dr)
        self.dr_dyn = (self.length) / self.n_r
        
        self.n_z = round(z_size / dz)
        self.dz_dyn = z_size / self.n_z
        
        # Center points of radial partitions, in m
        rr = np.linspace((R-self.length)+(self.dr_dyn/2), R-(self.dr_dyn/2), self.n_r)
        self.rr = rr*1e-6
        
        # Center points of z partitons, in m
        self.zz = np.linspace(-z_size/2 + self.dz_dyn/2, z_size/2 - self.dz_dyn/2, self.n_z)*1e-6
        
    def build(self, density_profile):
        
        # density_profile is function that returns density when passed (r,phi,z)
        # working in kms
        # for linear attractor, restrain phi to +- pi/8
        
        self.density_profile = density_profile
        z = 0. # considering cylindrical symmetry for now, can add wrapper for loop in z later
        self.sum_dm = 0.
        
        # dr, dz in um: convert
        dr = self.dr_dyn*1e-6
        dz = self.dz_dyn*1e-6
        
        for r in tqdm(self.rr, desc='Building Attractor'):
            
            radial_partition = {}
            
            # create angular partition, r in m

            dphi_finger = 0.003125
            
            n_phi = round(10*dphi_finger*(r*1e6) / self.dphi_i) # dphi_i given in microns, so convert r back to microns
            n_phi = int(n_phi)
            dphi_dyn = 10*dphi_finger / n_phi # in rads
            pp_upper = np.linspace(dphi_dyn/2, 10*dphi_finger - dphi_dyn/2, n_phi)
            pp_lower = 2*np.pi - np.flip(pp_upper)
            pp = np.concatenate((pp_upper, pp_lower))

            data = np.zeros((self.zz.size, pp.size))
            dV = r*dr*dphi_dyn*dz

            for i,phi in enumerate(pp):
                rho = density_profile(r, phi, z) # in kg/m^3
                dm = rho*dV # kg
                data[:,i] = dm
                self.sum_dm += dm

            radial_partition = {'params': (self.dr_dyn, dphi_dyn, self.dz_dyn, pp),
                                   'data': data}

            self.data[r] = radial_partition
        
        self.is_built = True
        # only for uniform test
        # self.sum_dm *= len(self.zz)

        
    def newtonian(self, rb_vec, Rb=2.32e-6, rhob=1550.0, debug=False):
        
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
    
    
    def yukawa(self, rb_vec, a=1., l=10e-6, Rb=2.32e-6, rhob=1550.0):
        
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
    
lin = attractor_profile()
lin.build(linear_fingers)

dphi_bead = dphi_finger/5
phi_bead_upper = np.arange(dphi_bead, 10*dphi_finger+dphi_bead, dphi_bead)
phi_bead_lower = 2*np.pi - np.flip(phi_bead_upper)
phi_bead_upper = np.concatenate(([0], phi_bead_upper))
phi_bead = np.concatenate((phi_bead_upper, phi_bead_lower))

seps = np.array([10, 20, 100])
heights = np.array([-5, 0, 5])

def linear_simulation(params):
    sep, phi, z = params
    
    fname = 'simdata/sep_'+str(sep)
    fname += '_phi_'+str(phi)
    fname += '_height_'+str(z)
    
    newt = lin.newtonian((R+sep, phi, z))
    yuka = lin.yukawa((R+sep, phi, z))
    
    try:
        np.save(fname+'_newt.npy', newt)
        np.save(fname+'_yuka.npy', yuka)
    except:
        print('Save Error: {}'.format(fname))
    
    return fname


param_list = list(itertools.product(seps, phi_bead, heights))
results = Parallel(n_jobs=20)(delayed(linear_simulation)(param) for param in tqdm(param_list))