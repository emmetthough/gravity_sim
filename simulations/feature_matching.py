import numpy as np
import os, sys
import dill as pickle
import time
from tqdm import tqdm

from joblib import Parallel, delayed
import itertools

from symmetric_attractor_profile import attractor_profile
print(os.getcwd())

ncores = 24
os.chdir('..')
parent = os.getcwd()
data_root = parent+'/sim_data/feature_matching_new/'
save_img = False
verbose = True

# define bead properties
Rb = 7.6/2
rhob = 1850.

# define attractor properties
R = 1000
height = 20
rho_attractor = 19300. # gold
# rho_attractor = 8900. # nickel

# seps = np.linspace(7, 150, 10)
# heights = np.linspace(-250, 250, 10)
seps = np.array([5, 20])
heights = np.array([5])
lambdas = np.array([1e-6, 10e-6])


def nholes(r,phi,z,R=R,hr=25,from_edge=2.5):
    # density profile for evenly spaced holes
    R *= 1e-6
    hr *= 1e-6
    from_edge *= 1e-6
    
    # outside attractor or inside bulk
    if r > R or r < (500e-6):
        return 0.
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    if (x-(R-(hr+from_edge)))**2 + y**2 <= hr**2: # inside the hole
        return 0.
    
    return rho_attractor


def find_attractor(results_dir, root=data_root):
    # look for attractor profile if it already exists
    try:
        os.chdir(root+results_dir)
        with open('attractor_profile.p', 'rb') as profile:
            attractor = pickle.load(profile)
            print('Attractor profile found and loaded')
            print()
            return attractor
    except:
        os.mkdir(root+results_dir)
        os.chdir(root+results_dir)
        print('No attractor profile found, building now')
        return None


def build_attractor(results_dir, N, hr, dist_from_edge, root=data_root):
    # pretty self-explainatory
    if os.getcwd() != root+results_dir:
        os.chdir(root+results_dir)
    
    attractor = attractor_profile(R, height, Rb=Rb, rhob=rhob)
    attractor.build(lambda pos: nholes(pos[0], pos[1], pos[2], R=R, hr=hr, from_edge=dist_from_edge), N, cyl_sym=True) # lambda to pass other params
    
    print()
    if save_img:
        attractor.plot_xy(save=True)
        print('Attractor Image Saved')
        print()
    pickle.dump(attractor, open('attractor_profile.p', 'wb'))
    return attractor
    
        
# define function to save the force arrays at a given r, z; for use with Parallelization
def rotary_simulation(params):
    # unpack params
    sep, height, results_path = params
    
    os.chdir(results_path)
    with open('attractor_profile.p', 'rb') as profile:
        attractor = pickle.load(profile)

    # create results dictionary
    results = {}
    results[sep] = {}
    results[sep][height] = {}
    results[sep][height]['newtonian'] = []
    results[sep][height]['yukawa'] = [lambdas, []]

    results['params'] = {}
    results['params']['rbead'] = Rb
    results['params']['rhobead'] = rhob

    # create filename
    fname  = 'sep_' + str(sep)
    fname += '_height_' + str(height)
    fname += '.p'
    full_fname = os.path.join(results_path, fname)

    # get force curves (we love abstraction!)
    comp_start = time.time()
    
    nphi = 2000//attractor.N # for comparison with full attractor data, will eventually be determined by spacing and separation
    phis, newt_forces, yuka_forces = attractor.full_gravity(sep, height, lambdas=lambdas, nphi=nphi)

    comp_end = time.time()
    comp_time = comp_end-comp_start

    # put force curves in results dictionary
    results[sep]['phis'] = phis
    results[sep][height]['newtonian'] = newt_forces
    results[sep][height]['yukawa'][1] = yuka_forces

    # save dat dict bro
    try:
        pickle.dump(results, open(full_fname, 'wb'))
    except:
        print('Save error! : ', full_fname)

    # return comp times for statistics
    return comp_time


def sim_main(params):
    # for use with Parallelization on different density profiles
    
    N, hr, dist_from_edge = params
    
    results_dir = 'N=' + str(N) + '_hr=' + str(hr) + '_from_edge=' + str(dist_from_edge)
    
    attractor = find_attractor(results_dir)
    if attractor == None:
        attractor = build_attractor(results_dir, N, hr, dist_from_edge)
    
    # basically just make a giant list from two giant lists
    param_list = list(itertools.product(seps, heights, [data_root+results_dir]))
    
    print()
    print('Starting simulation...')
    print()
    
    # for a given geometry, Parallelize if computing grid of sep, heights
    # otherwise makes more sense to Parallelize density profiles themselves
    sim_start = time.time()
    comp_times = Parallel(n_jobs=1)(delayed(rotary_simulation)(param) for param in tqdm(param_list))
    sim_end = time.time()
    
    pickle.dump(comp_times, open('times_raw.p', 'wb'))
    
    os.chdir(parent)
        
    if verbose:
    
        print()
        print('Simulation complete!')
        print()
        print('Statistics:')
        print()
        print('Total time: {:0.1f} s'.format(sim_end-sim_start))
        print(f'Mean: {np.mean(comp_times):0.1f} s, Std: {np.std(comp_times):0.1f} s per bead position (all phis and all lambdas)')

    
if __name__ == '__main__':
    # bigggg python boilerplate guy
    
    Ns = np.array([7])
    hrs = np.linspace(2.5, 25, 7)
    dist_from_edge = np.linspace(2.5, 25, 7)
    
    param_list = list(itertools.product(Ns, hrs, dist_from_edge))
    
    full_start = time.time()
    Parallel(n_jobs=25)(delayed(sim_main)(param) for param in param_list)
    full_end = time.time()
    
    print(f'Full sim took {full_end-full_start:.2f} hrs')