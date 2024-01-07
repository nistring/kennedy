import numpy as np
import casadi as ca
from scipy.optimize import minimize
from track_irl import *


def reward_function(parameters, state, input):
    feature1 = - state[0]
    feature2 = state[2] **2
    feature3 = input[0] **2
    feature4 = input[1] **2

    return np.dot(parameters, [feature1, feature2, feature3, feature4])

# Define a loss function for IRL to be minimized
def irl_loss(parameters, history_data):

    total_loss = 0.0

    for trajectory in history_data:
        predicted_rewards = np.array([reward_function(parameters, state, input) for state, input in zip(trajectory[:, :4], trajectory[:, 4:])])
        total_loss += np.mean((predicted_rewards - np.ones_like(predicted_rewards)) ** 2)
    
    # Entropy term (assuming a discrete action space for simplicity)
    policy_probs = np.exp(predicted_rewards) / np.sum(np.exp(predicted_rewards), axis=0, keepdims=True)
    entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))  # Add a small epsilon to avoid log(0)

    # Combine matching behavior and entropy terms
    total_loss /= len(history_data)
    total_loss -= 0.1 * entropy   
    
    return total_loss


    # return total_loss / len(history_data)


# Reformulate the optimized_traj
center_dir = './track/center_traj_with_boundary.txt' 
center = np.loadtxt(center_dir, delimiter=",", dtype = float)
for j in range(0,50):
    history = np.loadtxt('./data/optimized_traj'+str(j)+'.txt', delimiter=",", dtype = float)
    history_track = Track_irl(center, history)
    np.savetxt('./data/optimized_traj_'+str(j)+'.txt', history_track.n_data, delimiter=",")
    print("Trajectory"+str(j)+" is saved")


## Read trajectory information
history_arr = []
for j in range(50):
    history = np.loadtxt('./data/optimized_traj_'+str(j)+'.txt', delimiter=",", dtype = float)

    n_history = history.copy()
    for k in range(len(history)-1):
        n_history[k,0] = history[k+1,0] - history[k,0]
        n_history[k,4] = history[k+1,4] - history[k,4]
        n_history[k,5] = history[k+1,5] - history[k,5]

    if not np.isnan(history).any():
        history_arr.append(n_history[:-1])



# Learn reward function parameters using IRL
# Define the symbolic variables

initial_parameters = np.array([1, 0.5, 0.1, 0.1])
irl_result = minimize(irl_loss, initial_parameters, args=(history_arr,), method='SLSQP')
learned_parameters = irl_result.x

    
print(learned_parameters)
