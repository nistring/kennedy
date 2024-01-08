import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import casadi as ca

from track import * 
from PID import *

from optimize_ import *



''' Initialize the environment '''

## Set the directory
dir = './'# os.path.expanduser('~') + '/Desktop/IterativeTrackOptimization/'
center_dir = dir + 'track/center_traj_with_boundary.txt' 
inner_dir = dir + 'track/innerwall.txt'
outer_dir = dir + 'track/outerwall.txt'

## Read trajectory information
center = np.loadtxt(center_dir, delimiter=",", dtype = float)
inner = np.loadtxt(inner_dir, delimiter=",", dtype = float)
outer = np.loadtxt(outer_dir, delimiter=",", dtype = float)

Nsim = 1
for i in range(0,Nsim):
    # idx = np.random.randint(0,len(center))
    idx = 0
    new_center = center[idx:]
    new_center = np.concatenate((new_center,center[:idx]),axis=0)
    track = Track(new_center,inner,outer)
    model = KinematicBicycle(track, N = 200)
    ego_history, ego_sim_state, egost_list, = run_pid_warmstart(track, model, t=0.0)

    mpcc_ego_controller = Optimize(model, track, opt_params)
    mpcc_ego_controller.set_warm_start(*ego_history)

    q,u,states  = mpcc_ego_controller.solve_optimization(ego_sim_state)

    if q[-1,0] >= track.track_length:
        idx_s = np.where(q[:,0] >= track.track_length)[0][0]
    else:
        idx_s = -1

    if q[-1,0] >= track.track_length*2:
        idx_s2 = np.where(q[:,0] >= track.track_length*2)[0][0]
    else:
        idx_s2 = -1

    points = states[:idx_s,:2].reshape(-1, 1, 2)
    speed = q[:idx_s,3]

    ## Save the optimized trajectory
    traj = np.zeros((len(speed),6))
    traj[:,:2] = states[:idx_s,:2]
    traj[:,2] = states[:idx_s,2]
    traj[:,3] = q[:idx_s,3]
    traj[:,4:] = u[:idx_s]

    # np.savetxt('./data/optimized_traj'+str(i)+'.txt',traj, delimiter=",")
    np.savetxt('./data/optimized_traj.txt',traj, delimiter=",")

    fig = plt.figure()
    ax = plt.gca()
    ax.axis('equal')
    plt.plot(track.x, track.y,'--k')
    plt.plot(inner[:,0],inner[:,1], 'k')
    plt.plot(outer[:,0],outer[:,1], 'k')
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(speed), max(speed))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(speed)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)


    # points2 = states[idx_s:idx_s2,:2].reshape(-1, 1, 2)
    # speed2 = q[idx_s:idx_s2,3]

    # ## Save the optimized trajectory
    # traj = np.zeros((len(speed2),4))
    # traj[:,:2] = states[idx_s:idx_s2,:2]
    # traj[:,2] = states[idx_s:idx_s2,2]
    # traj[:,3] = q[idx_s:idx_s2,3]

    # np.savetxt('./data/optimized_traj_.txt',traj, delimiter=",")

    # fig = plt.figure()
    # ax = plt.gca()
    # ax.axis('equal')
    # plt.plot(track.x, track.y,'--k')
    # plt.plot(inner[:,0],inner[:,1], 'k')
    # plt.plot(outer[:,0],outer[:,1], 'k')
    # segments = np.concatenate([points2[:-1], points2[1:]], axis=1)
    # norm = plt.Normalize(min(speed2), max(speed2))
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # lc.set_array(speed2)
    # lc.set_linewidth(5)
    # line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)

    plt.show()
    print("Trajectory Done")