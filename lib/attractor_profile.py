import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from numpy.fft import fft, fftfreq

class attractor_profile:
    
    def __init__(self, R, z_size=1, N=10, dr=1, dz=1, dphi=1, r_range=1.02, Rb=5, rhob=1850., depth=None):
        
        # give all params in microns
        
        self.is_built = False
        self.data = {}
        self.dphi_i = dphi # in um
        self.R = R # in um
        self.N = N
        
        self.Rb = Rb*1e-6
        self.rhob = rhob
        
        # create partitions
        
        if depth is None or depth == R:
        
            self.n_r = int(round((R - 0.5*dr) / dr))
            self.dr_dyn = (R - 0.5*dr) / self.n_r
            
            # Center points of radial partitions, in m
            rr = np.linspace(self.dr_dyn, R-(self.dr_dyn/2), self.n_r)
            self.rr = np.concatenate(([0], rr))*1e-6
            
        else:
            
            self.n_r = int(depth // dr)
            self.dr_dyn = depth / self.n_r
            self.rr = np.linspace((R-depth) + (self.dr_dyn/2), R-(self.dr_dyn/2), self.n_r)*1e-6
        
        
        self.n_z = int(round(z_size / dz))
        self.dz_dyn = z_size / self.n_z
        
        # Center points of z partitons, in m
        self.zz = np.linspace(-z_size/2 + self.dz_dyn/2, z_size/2 - self.dz_dyn/2, self.n_z)*1e-6
        
    def build(self, density_profile, cyl_sym=False):
        
        # density_profile is function that returns density when passed (r,phi,z)
        # working in kms
        
        self.cyl_sym = cyl_sym
        self.density_profile = density_profile
        z = 0. # considering cylindrical symmetry for now, can add wrapper for loop in z later
        self.sum_dm = 0.
        self.n_pts = 0
        
        # dr, dz in um: convert
        dr = self.dr_dyn*1e-6
        dz = self.dz_dyn*1e-6
        
        for r in tqdm(self.rr, desc='Building Attractor'):
            
            radial_partition = {}
            
            # create angular partition, r in m
            if r != 0.:
                
                n_phi = int(round(2*np.pi*(r*1e6) / self.dphi_i)) # dphi_i given in microns, so convert r back to microns
                dphi_dyn = 2*np.pi / n_phi # in rads
                pp = np.linspace(dphi_dyn/2, 2*np.pi - dphi_dyn/2, n_phi)
                            
                data = np.zeros((self.zz.size, n_phi))
                dV = r*dr*dphi_dyn*dz
                
                if not self.cyl_sym:
                
                    for i,z in enumerate(self.zz):
                        for j,phi in enumerate(pp):
                            rho = density_profile((r, phi, z)) # in kg/m^3
                            dm = rho*dV # kg
                            data[i,j] = dm
                            self.sum_dm += dm
                            self.n_pts += 1
                else:
                    
                    for j,phi in enumerate(pp):
                            rho = density_profile((r, phi, z)) # in kg/m^3
                            dm = rho*dV # kg
                            data[:,j] = dm
                            self.sum_dm += dm*self.zz.size
                            self.n_pts += 1*self.zz.size

                radial_partition = {'params': (self.dr_dyn, dphi_dyn, self.dz_dyn, pp),
                                       'data': data}

            else: # r == 0
                
                data = np.zeros(self.zz.size)
                dV = np.pi*(dr/2)**2 * dz # m^3
                
                for i,z in enumerate(self.zz):
                    rho = density_profile((0.,0.,z)) # kg/m^3
                    dm = rho*dV  # kg
                    data[i] = dm
                    self.sum_dm += dm
                    self.n_pts += 1
                    
                radial_partition = {'params': (self.dr_dyn/2, 2*np.pi, self.dz_dyn, np.array([0.])), 
                                 'data': data} # phi value for center? None?

            self.data[r] = radial_partition
        
        self.is_built = True
        self.build_numpy()
        
    def build_numpy(self):
        
        # converts from dictionary structure (convenient for storing metadata) to numpy arrays (faster indexing in computations)
        
        # get biggest data list
        max_dm_size = self.data[self.rr[-1]]['data'].shape[1]
        
        # initialize big arrays, including 'sizes' used for indexing
        self.mass_arr = np.zeros((self.rr.size, self.zz.size, max_dm_size))
        self.phis_arr = np.zeros((self.rr.size, max_dm_size))
        self.sizes = np.zeros(self.rr.size, dtype=np.int16)
        
        # build 'em boyyyyy
        i = -1
        for r, partition in self.data.items():     
            i += 1
            
            dm_arr = partition['data'] # shape??
            if len(dm_arr.shape) == 1:
                dm_arr = np.reshape(dm_arr, (dm_arr.size, 1))
            pp = partition['params'][3]
            size = int(pp.size)
            
            self.sizes[i] = size
            self.phis_arr[i,:size] = pp
            self.mass_arr[i,:,:size] = dm_arr
        
    def plot_xy(self, downsample=50, upsample=2, show_data=False, save=False, ret=False):
        
        # convert angular data to cartesian and display
        # converting back to um for ease
        
        rr = np.array([0.])
        pp = np.array([0.])
        mm = np.array([self.data[0.]['data'][0]])
        
        xxi = np.array([0.])
        yyi = np.array([0.])
        
        # create full coordinate list of r, phi, rho
        for r in self.data.keys():
            
            partition = self.data[r]
            data_arr = partition['data']
            pp_i = partition['params'][3][::downsample]
            if pp_i.size != 1:
                dms = data_arr[0,::downsample]

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
        
        for i in tqdm(np.arange(xx.size), desc='Building Image'):
            
            x = xx[i]
            for j,y in enumerate(yy):
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                if phi < 0:
                    phi += 2*np.pi

                rho_cart[j,i] =  self.density_profile((r*1e-6,phi,0.)) #check index order
                
        fig, ax = plt.subplots(1, figsize=(12,12))   
        ax.contourf(xx,yy,rho_cart)
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        
        if show_data:
            for i in tqdm(np.arange(xxi.size), desc='Populating Image'):
                mi = mm[i]
                if mi != 0.:
                    c='r'
                else:
                    c='b'
                ax.scatter(xxi[i], yyi[i], c=c)
        
        if save:
            plt.savefig('attractor_profile.png', dpi=150)
        else:
            plt.show()
            
        if ret:
            return fig,ax
            
        
    def newtonian(self, rb_vec, verbose=False):
        
        # rb_vec given in um, tuple
        
        G = 6.67e-11
        Fg = np.array([0.,0.,0.])
        rb, pb, zb = rb_vec
        rb = rb*1e-6
        zb = zb*1e-6
        
        i = -1
        
        for i,r in enumerate(self.rr):      
            
            dm_arr = self.mass_arr[i,:,:self.sizes[i]].T # transpose here for shape, fix in build attractor?
            pp = self.phis_arr[i,:self.sizes[i]]
            
            if np.sum(dm_arr) != 0.:
                # create coordinate arrays to compute force vector at
                psep, zsep = np.meshgrid(pp - pb, self.zz - zb, indexing='ij')
                rsep = rb - r

                # separation between attractor mesh and bead
                sep = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep) + zsep**2)

                # separation in only r,phi
                sep_rp = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep))

                if r == 0.:
                    sep = np.reshape(sep, dm_arr.shape)

                # get full vector force at every point in meshgrid (i.e. every r,phi,z in partition)
                full_vec_force = -1.0 * (4.*G*dm_arr*self.rhob*np.pi*self.Rb**3)/(3.*sep**2)

                # get projections onto each unit vector and sum the force at all points on the partition
                Fg[0] += np.sum(full_vec_force * (rb-r*np.cos(psep))/sep_rp)
                Fg[1] += np.sum(full_vec_force * (r*np.sin(psep))/sep_rp)
                Fg[2] += np.sum(full_vec_force * (zsep/sep))   

        return Fg
    
    
    def yukawa(self, rb_vec, l=100e-6):
        
        # given bead position rb_vec, compute yukawa force
        
        G = 6.67e-11
        Fg = np.array([0.,0.,0.])
        
        rb, pb, zb = rb_vec
        rb = rb*1e-6
        zb = zb*1e-6
        
        func = np.exp(-2. * self.Rb/l) * (1. + self.Rb/l) + self.Rb/l - 1.
        
        for i,r in enumerate(self.rr):      
            
            dm_arr = self.mass_arr[i,:,:self.sizes[i]].T # transpose here for shape, fix in build attractor?
            pp = self.phis_arr[i,:self.sizes[i]]
            
            if np.sum(dm_arr) != 0.:
                
                # create coordinate arrays to compute force vector at
                psep, zsep = np.meshgrid(pp - pb, self.zz - zb, indexing='ij')
                rsep = rb - r

                # separation between attractor mesh and SURFACE of bead
                sep = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep) + zsep**2) - self.Rb

                # separation in only r,phi; to center for projections
                sep_rp = np.sqrt((rb**2) + (r**2) - 2*rb*r*np.cos(psep))

                # yukawa term
                prefac = -((2.*G*dm_arr*self.rhob*np.pi) / (3.*(sep+self.Rb)**2))
                yukterm = 3.*l**2 * (sep+self.Rb+l) * func * np.exp(-sep/l)

                # get full vector force at every point in meshgrid (i.e. every r,phi,z in partition)
                full_vec_force = prefac * yukterm

                # add 1% white noise
                # full_vec_force += np.random.randn()*full_vec_force/100

                # get projections onto each unit vector and sum the force at all points on the partition
                Fg[0] += np.sum(full_vec_force * (rb-r*np.cos(psep))/sep_rp)
                Fg[1] += np.sum(full_vec_force * (r*np.sin(psep))/sep_rp)
                Fg[2] += np.sum(full_vec_force * (zsep/(sep+self.Rb)))               
        
        return Fg      
    
    
    def full_gravity(self, sep, height, lambdas=None, nphi=5):
        
        phis = np.linspace(0, 2*np.pi, nphi)
        
        newt = np.zeros((nphi, 3))
        yuka = np.zeros((nphi, lambdas.size, 3))
        
        for i,phi in enumerate(phis):
            newt[i,:] = self.newtonian((self.R+sep, phi, height))
            for j,lam in enumerate(lambdas):
                yuka[i,j,:] = self.yukawa((self.R+sep, phi, height), l=lam)
        
        return phis, newt, yuka
            