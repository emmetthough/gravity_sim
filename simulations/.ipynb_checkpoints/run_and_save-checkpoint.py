import numpy as np
import os, sys
import dill as pickle
import time
from tqdm import tqdm

from joblib import Parallel, delayed
import itertools

from attractor_profile import attractor_profile

ncores = 24
parent = os.getcwd()
data_root = parent+'/sim_data/'
save_img = True
verbose = True

# define bead properties
Rb = 7.6/2
rhob = 1850.


# define attractor properties
R = 1000
height = 20
rho_attractor = 19300. # gold
# nickel 8900.

# seps = np.linspace(7, 150, 10)
# heights = np.linspace(-250, 250, 10)
seps = np.array([20])
heights = np.array([5])
phis = np.linspace(0, 2*np.pi, 2000)
lambdas = np.array([10e-6, 50e-6])

print('{} seps, {} heights, {} phis, {} lambdas'.format(seps.size, heights.size, phis.size, lambdas.size))
print()


# density profile for evenly spaced holes
def nholes(r,phi,z,R=R,hr=25,Rholes=R-50, N=10):
    R *= 1e-6
    if r > R:
        return 0.
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    
    dphi = 2*np.pi/N
    hphis = dphi*np.arange(N)
    
    hr *= 1e-6
    
    # define center of holes
    hxs = Rholes*1e-6*np.cos(hphis)
    hys = Rholes*1e-6*np.sin(hphis)
    
    for hx,hy in zip(hxs, hys):
        if (x-hx)**2 + (y-hy)**2 <= hr**2: # inside a hole
            return 0.
    
    return rho_attractor


def modulated_holes(r,phi,z,R=R,N=5):
    
    R *= 1e-6
    
    if r > R or r < 500e-6: # 500um bulk
        return 0.
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    
    dphi = 2*np.pi/N
    Hphis = dphi*np.arange(N)
    
    # center of big holes, radius 25um
    Hxs = (R-(30+25)*1e-6)*np.cos(Hphis)
    Hys = (R-(30+25)*1e-6)*np.sin(Hphis)
    
    # in bigger holes
    for Hx,Hy in zip(Hxs, Hys):
        if (x-Hx)**2 + (y-Hy)**2 <= (25e-6)**2:
            return 0.
    
    hphis_upper = Hphis+0.05076
    hphis_lower = Hphis-0.05076
    
    hphis = np.concatenate((hphis_upper, hphis_lower))
    hphis[hphis < 0] += 2*np.pi
    hphis = np.concatenate((Hphis, hphis))
    
    # center of small holes, radius 12.5um
    hxs = (R-(12.5+2.5)*1e-6)*np.cos(hphis)
    hys = (R-(12.5+2.5)*1e-6)*np.sin(hphis)
    
    for hx,hy in zip(hxs, hys):
        if (x-hx)**2 + (y-hy)**2 <= (12.5e-6)**2:
            return 0.
    
    return rho_attractor

def find_attractor(results_dir, root=data_root):
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
    
    if os.getcwd() != root+results_dir:
        os.chdir(root+results_dir)
    
    attractor = attractor_profile(R, height, N=N, Rb=Rb, rhob=rhob)
    
    Rholes = R - (dist_from_edge+hr)
    # attractor.build(lambda pos: nholes(pos[0], pos[1], pos[2], N=N, hr=hr, Rholes=Rholes), cyl_sym=True)
    attractor.build(lambda pos: modulated_holes(pos[0], pos[1], pos[2], N=5), cyl_sym=True)
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
    results[sep]['phis'] = phis # redundant for now, might be useful if dynamic in future
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

    # initialize lists to add to for each phi location
    newt_forces = []
    yuka_forces = []

    newt_times = []
    yuka_times = []

    comp_start = time.time()

    for phi in phis:
        bead_pos = (R+sep, phi, height)

        # compute Newtonian force in cylindrical coords wrt the bead
        newt_start = time.time()
        newt_forces.append(attractor.newtonian(bead_pos))
        newt_end = time.time()

        newt_times.append(newt_end-newt_start)

        # compute Yukawa force in cylindrical coords wrt the bead for each yukawa lambda parameter
        yuka_lambdas = []
        for lambda_i in lambdas:
            yuka_start = time.time()
            yuka_lambdas.append(attractor.yukawa(bead_pos, l=lambda_i))
            yuka_end = time.time()

            yuka_times.append(yuka_end-yuka_start)

        yuka_forces.append(yuka_lambdas)

    comp_end = time.time()
    comp_time = comp_end-comp_start

    # put force curves in results dictionary
    results[sep][height]['newtonian'] = newt_forces
    results[sep][height]['yukawa'][1] = yuka_forces

    # save dat dict bro
    try:
        pickle.dump(results, open(full_fname, 'wb'))
    except:
        print('Save error! : ', full_fname)

    # return comp times for statistics
    return [comp_time,newt_times,yuka_times]


def sim_main(params):
    
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
    
    # parallelize the f outta da sim yo and pray it doesn't fail or take the age of the universe
    sim_start = time.time()
    times = Parallel(n_jobs=1)(delayed(rotary_simulation)(param) for param in tqdm(param_list))
    sim_end = time.time()
    
    pickle.dump(times, open('times_raw.p', 'wb'))
    
    os.chdir(parent)
    
    comp_times = []
    newt_times = np.array([])
    yuka_times = np.array([])
    
    for i in range(len(times)):
        times_i = times[i]
        comp_times.append(times_i[0])
        newt_times = np.concatenate((newt_times,np.array(times_i[1])))
        yuka_times = np.concatenate((yuka_times,np.array(times_i[2])))
        
    if verbose:
    
        print()
        print('Simulation complete!')
        print()
        print('Statistics:')
        print()
        print('Total time: {:0.1f} s'.format(sim_end-sim_start))
        print('Mean: {:0.1f} s, Std: {:0.1f} ms per bead position (all phis and all lambdas)' \
               .format(np.mean(comp_times), np.std(comp_times)*1e3))
        print('Newtonian mean: {:0.1f} s, std: {:0.1f} s'.format(np.mean(newt_times), np.std(newt_times)))
        print('Yukawa mean: {:0.1f} s, std: {:0.1f} s'.format(np.mean(yuka_times), np.std(yuka_times)))
        print('With N={} phi and N={} lambdas, expect one sim call to take {:0.1f} s' \
              .format(phis.size, lambdas.size, phis.size*(np.mean(newt_times)+lambdas.size*np.mean(yuka_times))))
        print()
    
if __name__ == '__main__':
    
    sim_main((5, 12.5, 2.5))
    
#     dist_from_edge = np.array([2.5])
#     Ns = np.array([5])
#     hrs = np.array([12.5])
    
#     param_list = list(itertools.product(Ns, hrs, dist_from_edge))
#     Parallel(n_jobs=1)(delayed(sim_main)(param) for param in param_list)