import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import os
import scipy.interpolate as intp
plt.rcParams['figure.figsize'] = (12,8)

from attractor_profile import attractor_profile
import bead_util as bu

parent = os.getcwd()

class holes_data:
    # class for reading simulation data and building interpolating functions
    
    def __init__(self, data_dir='/sim_data/'):
        self.data_dir = data_dir
        self.raw_data = self.process_raw_data()
        self.build_params()
        self.data = self.build_Gfuncs()
    
    def get_dirs(self):
        # wrapper for getting dir names in data dir
        return [fold for fold in os.listdir(self.data_dir) if fold.split('=')[0]=='N']

    def process_raw_data(self, verbose=False):
        # loads raw sim data into a dictionary with specified order
        
        data = {}
        data['order'] = '[N][from_edge][hr][sep][height]'

        Ns = []
        hrs = []
        from_edges = []
        
        # loop through the different parameter trials
        for Ndir in self.get_dirs():
            
            os.chdir(self.data_dir+Ndir)
            
            # extract params from directory name
            splits = Ndir.split('_')
            N = int(splits[0].split('=')[1])
            Ns.append(N)
            hr = float(splits[1].split('=')[1])
            hrs.append(hr)
            from_edge = float(splits[-1].split('=')[1])
            from_edges.append(from_edge)

            if verbose:
                print(Ndir)
                print('N=', N)
                print('hr=', hr)
                print('from_edge=', from_edge)
                print()
            
            # build the relevant subdictionaries if necessary
            if N not in data.keys():
                data[N] = {}
            if from_edge not in data[N].keys():
                data[N][from_edge] = {}
            if hr not in data[N][from_edge].keys():
                data[N][from_edge][hr] = {}
                
            # loop through the different seps and heights within each trial
            files = [file for file in os.listdir() if file.split('.')[-1]=='p'] # only pickle files
            files = [file for file in os.listdir() if len(file.split('_')) == 4] # only data files
            for file in files:
                # extract params from filename
                splits = file.split('_')
                sep = float(splits[1])
                height = float(splits[3].split('.')[0])
                # read in the data
                with open(file, 'rb') as f:
                    trial_dict = pickle.load(f)
                    # initialize with keys and phi data if necessary
                    if sep not in data[N][from_edge][hr].keys():
                        data[N][from_edge][hr][sep] = {}
                        data[N][from_edge][hr][sep]['phis'] = trial_dict[sep]['phis']
                    if height not in data[N][from_edge][hr][sep].keys():
                        data[N][from_edge][hr][sep][height] = {}
                    # parse force data, numpy it for ease of access
                    newt_arr = np.array(trial_dict[sep][height]['newtonian'])
                    lambdas = np.array(trial_dict[sep][height]['yukawa'][0])
                    yuka_arr = np.array(trial_dict[sep][height]['yukawa'][1])
                    # set vals in data dict
                    data[N][from_edge][hr][sep][height]['newtonian'] = newt_arr
                    data[N][from_edge][hr][sep][height]['lambdas'] = lambdas
                    data[N][from_edge][hr][sep][height]['yukawa'] = yuka_arr
            os.chdir(parent)

        data['keys_lst'] = [Ns, from_edges, hrs]

        return data
    
    def build_params(self):
        
        self.Ns = []
        self.hrs = []
        self.from_edges = []
        
        for N,from_edge,hr in zip(*self.raw_data['keys_lst']):
            if N not in self.Ns:
                self.Ns.append(N)
            if from_edge not in self.from_edges:
                self.from_edges.append(from_edge)
            if hr not in self.hrs:
                self.hrs.append(hr)
    
    def build_Gfuncs(self):
        
        # from the raw simulation force data, build interpolating functions to sample from
    
        data = self.raw_data.copy()
        
        # loop through all parameter space
        for N,from_edge,hr in zip(*data['keys_lst']):
            trial = data[N][from_edge][hr]
            for sep in trial.keys():
                phis = trial[sep]['phis']
                for height in [key for key in trial[sep].keys() if key != 'phis']:
                    
                    # grab force data
                    newt_data = trial[sep][height]['newtonian']
                    yuka_data = trial[sep][height]['yukawa']
                    lambdas = trial[sep][height]['lambdas']
                    
                    # newtonian interpolation
                    newt_interp = intp.interp1d(phis, newt_data.T, kind='cubic', axis=1)
                    
                    # yukawa interpolation for each value of lambda
                    yuka_interps = []
                    for i in np.arange(lambdas.size):
                        yuka_interp = intp.interp1d(phis, np.swapaxes(yuka_data[:,i,:], 0, 1), kind='cubic', axis=1)
                        yuka_interps.append(yuka_interp)

                    data[N][from_edge][hr][sep][height]['newt_funcs'] = newt_interp
                    data[N][from_edge][hr][sep][height]['yuka_funcs'] = yuka_interps
                    data[N][from_edge][hr][sep][height]['bounds'] = (np.min(phis), np.max(phis))

        return data
    
    
    
class holes_analysis:
    # class for doing analysis on a particular attractor geometry
    
    def __init__(self, Gfunc_data_dict, params):
        # Gfunc_data_dict output of holes_data
        # params = N,from_edge,hr,sep,height fully specifying attractor geometry
        
        self.data = Gfunc_data_dict
        self.params = params
        
        
    def sample_Gdata(self, w=1, tint=10, fsamp=5e3):
        # simulate real data as if the attractor were moving
        
        N,from_edge,hr,sep,height = self.params
        trial = self.data[N][from_edge][hr][sep][height]

        newt_funcs = trial['newt_funcs']
        yuka_funcs = trial['yuka_funcs']
        lambdas = trial['lambdas']
        lower, upper = trial['bounds']

        # how far the attractor moves between samples
        dphi = (upper-lower)*w/fsamp
        phis = np.arange(lower, upper, dphi)
        
        # make force curve for one full rotation
        newt_one = newt_funcs(phis)
        yuka_one = []
        for i in np.arange(lambdas.size):
            yuka_one.append(yuka_funcs[i](phis))
        yuka_one = np.array(yuka_one)
        
        # stack for subsequent revolutions within the integration period
        N = int(round(w*tint))
        n = newt_one.shape[1]
        newt = np.zeros((3, N*n))
        yuka = np.zeros((lambdas.size, 3, N*n))

        for i in np.arange(N):
            newt[:,n*i:n*(i+1)] = newt_one
            yuka[:,:,n*i:n*(i+1)] = yuka_one

        return np.arange(0, N/w, 1/fsamp), newt, (yuka, lambdas)

    def asd(self, force_data, fsamp=5e3):
        # get full r,phi,z ASD in units of N/root(Hz)
        
        fft = np.fft.rfft(force_data) 
        freqs = np.fft.rfftfreq(force_data.shape[1], d=1/fsamp)
        norm = np.sqrt(2.0*(freqs[1]-freqs[0]))*bu.fft_norm(force_data.shape[1], fsamp)

        asd = np.abs(norm*fft)

        return freqs, asd

    def signal_bins(self, psd_data, w=1, num_harmonics=10):
        # extract 

        freqs, psd = psd_data
        freq_inds = []
        harmonic_freqs = []
        psd_vals = []
        
        N = self.params[0]

        for i in np.arange(1, num_harmonics):
            harmonic_freqs.append(N*w*i)
            freq_ind = np.where(freqs==N*w*i)[0]
            freq_inds.append(freq_ind)
            if len(psd.shape) == 1:
                psd_vals.append(psd[freq_ind])
            else:
                psd_vals.append(psd[:,freq_ind])

        return freq_inds, harmonic_freqs, np.array(psd_vals)
    
    def sum_harmonics(self, w=1, tint=10, fsamp=5e3, num_harmonics=10, verbose=False):
        
        # return array of summed harmonic signals
        # 'x' axis: axis of signal (radial, angular, axial)
        # 'y' axis: signal type (newtonian, yukawa)
        
        N = self.params[0]
        times, newtsamp, (yukasamp, lambdas) = self.sample_Gdata(w=w, tint=tint, fsamp=fsamp)
        
        freqs, newtasd = self.asd(newtsamp, fsamp=fsamp)
        inds, sigfreqs, nsignals = self.signal_bins((freqs, newtasd), w=w, num_harmonics=num_harmonics)
                
        yukasignals = []
        for i in np.arange(yukasamp.shape[0]):
            _, yukaasd = self.asd(yukasamp[i], fsamp=5e3)
            _, _, yukasignal = self.signal_bins((freqs, yukaasd), w=w, num_harmonics=num_harmonics)
            yukasignals.append(yukasignal)
            
        harmonic_sum = np.zeros((1+lambdas.size, 3))
        harmonic_sum[0,:] = np.sum(nsignals[:,:,0], axis=0)
        
        for i in range(lambdas.size):
            harmonic_sum[1+i, :] = np.sum(yukasignals[i][:,:,0], axis=0)
            
        if verbose:
            print(f'First {num_harmonics} harmonics:')
            print()
            print('           Radial     Angular     Axial')
            print(f'Newtonian: {harmonic_sum[0,0]:.3e}  {harmonic_sum[0,1]:.3e}   {harmonic_sum[0,2]:.3e}')
            print('Yukawa:')
            for i,lam in enumerate(lambdas):
                print(f'l={lam/1e-6:.2f}um: {harmonic_sum[i+1,0]:.3e}  {harmonic_sum[i+1,1]:.3e}   {harmonic_sum[i+1,2]:.3e}')
            print()
            
        
        return harmonic_sum
    
    def plot_Gdata(self, w=1, tint=2, fsamp=5e3):
        
        times, newtsamp, (yukasamp, lambdas) = self.sample_Gdata(w=w, tint=tint, fsamp=fsamp)
        components = ['radial', 'angular', 'axial']
        
        fig,ax = plt.subplots(3, 1+lambdas.size, figsize=(12,12))
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        for i, component in enumerate(components):
            ax[i,0].plot(times, newtsamp[i], label=component)
            for j, lam in enumerate(lambdas):
                ax[i,1+j].plot(times, yukasamp[j][i], label=f'$\lambda$ = {lam/1e-6:.0f}$\mu$m')
                if i == 0:
                    ax[i, 1+j].set_title(f'Yukawa, $\lambda$ = {lam/1e-6:.0f} $\mu$m', fontsize=18)
                
            ax[i,0].legend(fontsize=12)
            ax[i,1].legend(fontsize=12)
            
            ax[2,0].set_xlabel('Time [s]', fontsize=18)
            ax[2,1].set_xlabel('Time [s]', fontsize=18)
            
            ax[1,0].set_ylabel('Force [N]', fontsize=18)
            
            ax[0,0].set_title('Newtonian', fontsize=18)
        
        return fig, ax
        
        
    def plot_asd(self, w=1, tint=10, fsamp=5e3, modulated=False):
        
        N, edge, hr, sep, height = self.params
        
        times, newtsamp, (yukasamp, lambdas) = self.sample_Gdata(w=w, tint=tint, fsamp=fsamp)
        
        freqs, newtasd = self.asd(newtsamp, fsamp=fsamp)
        sigfreqs_ind, sigfreqs, nsignals = self.signal_bins((freqs, newtasd), w=w)
        
        yukaasds = []
        yukasignals = []
        for i in np.arange(yukasamp.shape[0]):
            _, yukaasd = self.asd(yukasamp[i], fsamp=5e3)
            yukasignal = self.signal_bins((freqs, yukaasd), w=w)
            yukaasds.append(yukaasd)
            yukasignals.append(yukasignal)
            
        components = ['radial', 'angular', 'axial']
            
        fig1,ax1 = plt.subplots(3, 2, figsize=(12,12), sharey=True)
        shift = 0.25
        
        ax1[0,0].set_title('Newtonian', fontsize=18)
        ax1[0,1].set_title('Yukawa', fontsize=18)
        ax1[1,0].set_ylabel('Force ASD [N/$\sqrt{Hz}$]', fontsize=18)
        ax1[2,0].set_xlabel('Frequency [Hz]', fontsize=18)
        ax1[2,1].set_xlabel('Frequency [Hz]', fontsize=18)
        if not modulated:
            title = f'{N} isotropic {hr}um radius holes {edge}um from edge, \n (sep, height)=({sep}, {height}), $\omega$={w}Hz, fsamp={fsamp/1e3:.1f}kHz'
        else:
            title = f'{N} modulated {hr}um radius holes {edge}um from edge, \n (sep, height)=({sep}, {height}), $\omega$={w}Hz, fsamp={fsamp/1e3:.1f}kHz'
        plt.suptitle(title, fontsize=20)
        
        for i, component in enumerate(components):
            ax1[i,0].loglog(freqs, newtasd[i], label=component)
            ax1[i,0].scatter(w, newtasd[i][np.where(freqs==w)], color='g', marker='x', label='Drive')
            ax1[i,0].scatter(N*w, newtasd[i][np.where(freqs==N*w)], color='r', marker='x', label='Fundamental')
            mins = [newtasd[i][np.where(freqs==w)]]
            maxs = [np.max(newtasd[i])]
            for j, lam in enumerate(lambdas): 
                ax1[i,1].loglog(freqs+j*shift, yukaasds[j][i], alpha=0.7, label=component+ f', $\lambda$ = {lam/1e-6:.1f}um')
                ax1[i,1].scatter(w+j*shift, yukaasds[j][i][np.where(freqs==w)], color='g', marker='x')
                ax1[i,1].scatter(N*w+j*shift, yukaasds[j][i][np.where(freqs==N*w)], color='r', marker='x')
                
                mins.append(yukaasds[j][i][np.where(freqs==w)])
                maxs.append(np.max(yukaasds[j][i]))
                
            ax1[i,0].legend(fontsize=12, loc='upper right')
            ax1[i,1].legend(fontsize=12, loc='upper right')
            
            ax1[i,0].grid()
            ax1[i,1].grid()
                
            ax1[i,0].set_ylim(np.min(mins)/100, np.max(maxs)*100)
            ax1[i,1].set_ylim(np.min(mins)/100, np.max(maxs)*100)
            ax1[i,0].set_xlim(0.5, 500)
            ax1[i,1].set_xlim(0.5, 500)
            
        plt.subplots_adjust(wspace=0.05)
                
        for ax in ax1:
            for a in ax:
                a.grid()
                pass
        
        return fig1, ax1
    
    def plot_signals(self, w=1, tint=10, fsamp=5e3, num_harmonics=10, modulated=False, title='default', log=True):
        
        N, edge, hr, sep, height = self.params
        
        times, newtsamp, (yukasamp, lambdas) = self.sample_Gdata(w=w, tint=tint, fsamp=fsamp)
        
        freqs, newtasd = self.asd(newtsamp, fsamp=5e3)
        sigfreqs_ind, sigfreqs, nsignals = self.signal_bins((freqs, newtasd), w=w, num_harmonics=num_harmonics)
        
        yukaasds = []
        yukasignals = []
        for i in np.arange(yukasamp.shape[0]):
            _, yukaasd = self.asd(yukasamp[i], fsamp=5e3)
            _,_, yukasignal = self.signal_bins((freqs, yukaasd), w=w, num_harmonics=num_harmonics)
            yukaasds.append(yukaasd)
            yukasignals.append(yukasignal)
            
        
        components = ['radial', 'angular', 'axial']
            
        fig1,ax1 = plt.subplots(3, 1, figsize=(12,12), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.1)
        ax1[0].set_title(f'First {num_harmonics} Harmonics', fontsize=18)
        ax1[1].set_ylabel('Force ASD [N/$\sqrt{Hz}$]', fontsize=18)
        ax1[2].set_xlabel('Frequency [Hz]', fontsize=18)
        if not modulated:
            title = f'{N} isotropic {hr}um radius holes {edge}um from edge, \n (sep, height)=({sep}, {height}), $\omega$={w}Hz, fsamp={fsamp/1e3:.1f}kHz'
        elif title=='default':
            title = f'{N} modulated {hr}um radius holes {edge}um from edge, \n (sep, height)=({sep}, {height}), $\omega$={w}Hz, fsamp={fsamp/1e3:.1f}kHz'
        else:
            title = title
        plt.suptitle(title, fontsize=20)
        
        for i,component in enumerate(components):
            ax1[i].plot(sigfreqs, nsignals[:,i], 'o-', label=component+' newtonian')
            for j, lam in enumerate(lambdas):
                ax1[i].plot(sigfreqs, yukasignals[j][:,i], 'o-', label=component+ f', $\lambda$ = {lam/1e-6:.1f}um')
            ax1[i].set_xticks(sigfreqs)
            ax1[i].legend(fontsize=12, loc='upper right')
            ax1[i].grid()
            if log:
                ax1[i].set_yscale('log')
            
        return fig1, ax1
        