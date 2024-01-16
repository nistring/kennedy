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

Nsim = 2
for i in range(0,Nsim):
    # idx = np.random.randint(0,len(center))
    idx = 0
    new_center = center[idx:]
    new_center = np.concatenate((new_center,center[:idx]),axis=0)
    track = Track(new_center,inner,outer)
    curvature_info = track.get_curvature_steps(N=200)
    
    model = KinematicBicycle(track, N = 200)
    ego_history, ego_sim_state, egost_list, = run_pid_warmstart(track, model, t=0.0)
    # print(ego_sim_state, ego_sim_state.shape[0])
    if ego_sim_state.v <=0 : 
        ego_sim_state.v =0.5
        
    if i>= 1:
        ego_sim_state.v = speed[-1]
        print(ego_sim_state.v)
    mpcc_ego_controller = Optimize(model, track, opt_params)
    mpcc_ego_controller.set_warm_start(*ego_history)

    q,u,states  = mpcc_ego_controller.solve_optimization(ego_sim_state)
    print(states)
    optimized_s_values = q[:, 0]
    
    if optimized_s_values[-1] >= track.track_length:
        idx_s = np.where(optimized_s_values >= track.track_length)[0][0]
    else:
        idx_s = -1
        
    if optimized_s_values[-1] >= track.track_length * 2:
        idx_s2 = np.where(optimized_s_values >= track.track_length * 2)[0][0]
    else:
        idx_s2 = -1
    # if q[-1,0] >= track.track_length:
    #     idx_s = np.where(q[:,0] >= track.track_length)[0][0]
    # else:
    #     idx_s = -1

    # if q[-1,0] >= track.track_length*2:
    #     idx_s2 = np.where(q[:,0] >= track.track_length*2)[0][0]
    # else:
    #     idx_s2 = -1

    points = states[:idx_s,:2].reshape(-1, 1, 2)
    speed = q[:idx_s,2]

    ## Save the optimized trajectory
    traj = np.zeros((len(speed),6))
    traj[:,0] = optimized_s_values[:idx_s]
    traj[:,1:3] = states[:idx_s,:2]
    traj[:,3] = q[:idx_s,2]
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

    # print(q,states)
    # v_value=q[:,3:].flatten()
    # v_value = np.where(v_value==0,np.inf, v_value)
        
    # start_s = q[1,0]
    # print("start",start_s)
    # end_s= track.track_length
    # print("end", end_s)
    # step_s = end_s / model.N
    # print("Step,", step_s)
    # s_values = np.arange(start_s,end_s,step_s)
    # if s_values[-1] != end_s:
    #     s_values = np.append(s_values,end_s)

    # print("v_value shape:", v_value.shape)
    # print("s_values shape:", s_values.shape)
    # opt_t = np.trapz(1/v_value, s_values)
    # print("opt",opt_t)
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