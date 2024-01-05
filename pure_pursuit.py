import numpy as np
import matplotlib.pyplot as plt
from track_utils import *

class VelocityController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.error = 0.0
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_velocity,target_velocity):
        self.target_velocity = target_velocity
        self.error = self.target_velocity - current_velocity
        self.integral += self.error
        derivative = self.error - self.prev_error

        control_input = self.kp * self.error + self.ki * self.integral + self.kd * derivative

        self.prev_error = self.error

        return control_input

class PurePursuit:
    def __init__(self, lookahead_distance_ratio):
        self.lookahead_distance_ratio = lookahead_distance_ratio
        self.target_idx = -1

    def find_target_point(self, current_position, current_velocity, path_points):
        # Find the point on the path that is closest to the vehicle
        min_distance = float('inf')
        target_point = None
        lookahead_distance = lookahead_distance_ratio * current_velocity
        lookahead_distance = np.clip(lookahead_distance, 0.5,3)
        if self.target_idx <= len(path_points)-10:
            for i in range(self.target_idx, len(path_points)):
                distance = np.linalg.norm(current_position - path_points[i,:2])
                if distance < min_distance and distance > lookahead_distance :
                    min_distance = distance
                    target_point = path_points[i,:2]
                    self.target_idx = i
        else:
            for i in range(len(path_points)):
                distance = np.linalg.norm(current_position - path_points[i,:2])
                if distance < min_distance and distance > lookahead_distance :
                    min_distance = distance
                    target_point = path_points[i,:2]
                    self.target_idx = i
        
        # current_velocity
        

        return self.target_idx, target_point

    def calculate_steering_angle(self, current_position, current_heading, target_point):
        # Calculate the angle between the vehicle heading and the line connecting the vehicle to the target point
        delta_x = target_point[0] - current_position[0]
        delta_y = target_point[1] - current_position[1]

        alpha = np.arctan2(delta_y, delta_x) - current_heading

        # Calculate the steering angle using the geometric relationship
        L = np.linalg.norm([delta_x, delta_y])  # Distance between the vehicle and the target point
        steering_angle = np.arctan2(2 * 0.5 * np.sin(alpha), L)

        return steering_angle

    def plot_path_and_trajectory(self, path_points, vehicle_trajectory):
        plt.figure(2)
        plt.plot(*zip(*path_points), '--k', label='Path Points')
        plt.plot(*zip(*vehicle_trajectory), marker='x', label='Vehicle Trajectory')
        plt.title('Pure Pursuit Path Tracking')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        # plt.legend()
        plt.grid(True)
        

    def plot_path_and_point(self, path_points, point, target):
        plt.figure(1)
        plt.plot(*zip(*path_points), '--k', label='Path Points')
        plt.plot(point[0], point[1], color='r', marker='x', label='Vehicle Trajectory')
        plt.plot(target[0], target[1],color='b',  marker='*', label='lookahead')
        plt.title('Pure Pursuit Path Tracking')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        # plt.legend()
        plt.grid(True)
        plt.pause(0.01)
        plt.draw()

def simulate_pure_pursuit(path_points, initial_position, initial_heading, lookahead_distance_ratio, num_steps):
    pure_pursuit = PurePursuit(lookahead_distance_ratio)
    velocity_controller = VelocityController(kp=0.1,ki=0.01, kd=0)

    current_position = np.array(initial_position)
    current_heading = initial_heading
    current_velocity = 0.0
    prev_velocity = 0.0
    vehicle_trajectory = [[current_position[0],current_position[1], initial_heading, 0.0]]
    vehicle_input = [[0.0, 0.0]]
    dt = 0.1 
    time = 0.0

    for _ in range(num_steps):
        
        time += 1
        target_idx, target_point = pure_pursuit.find_target_point(current_position, current_velocity, path_points[:,:2])
        steering_angle = pure_pursuit.calculate_steering_angle(current_position, current_heading, target_point)

        # Simulate vehicle dynamics (for simplicity, assuming a point-mass model)
        # Update velocity using the PID controller
        velocity_input = velocity_controller.update(current_velocity, path_points[target_idx,3])
        current_velocity += velocity_input
        # current_velocity = current_velocity
    
        
         # Time step for simulation
        current_heading += current_velocity*np.tan(steering_angle) * dt
        current_position[0] += current_velocity * np.cos(current_heading) * dt + np.random.randn()*0.02
        current_position[1] += current_velocity * np.sin(current_heading) * dt + np.random.randn()*0.02

        if time >= 10 and np.linalg.norm(current_position - initial_position) <= 0.5:
            break
        current_state = [current_position[0], current_position[1], current_heading, current_velocity]
        vehicle_trajectory.append(current_state.copy())
        vehicle_input.append([(current_velocity-prev_velocity)/dt, steering_angle])
        prev_velocity = current_velocity
        # pure_pursuit.plot_path_and_point(path_points[:,:2], current_position, target_point)

    pure_pursuit.plot_path_and_trajectory(path_points[:,:2], np.array(vehicle_trajectory)[:,:2])

    return vehicle_trajectory, vehicle_input

def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle

# Example usage
path_points = np.loadtxt('./data/optimized_traj.txt', delimiter=",", dtype = float)
path_points = interpolate(path_points, 0.1)
path_points = np.concatenate((path_points,path_points), axis=0)

sim = 50
# idx = np.random.randint(0,len(path_points), size=sim)
for j in range(sim):
    
    print("Trajectory="+str(j))
    initial_position = path_points[0,:2]
    initial_heading = path_points[0,2]
    lookahead_distance_ratio = 0.1
    num_steps = 200

    traj, input = simulate_pure_pursuit(path_points, initial_position, initial_heading, lookahead_distance_ratio, num_steps)

    history = np.zeros((len(traj),6))
    history[:,:4] = traj[:len(traj)]

    dx = np.gradient(history[:,0])
    dy = np.gradient(history[:,1])
    psi = np.arctan2(dy,dx)
    for w in range(len(traj)):
        history[w,2] = wrap_angle(psi[w])

    history[:,4:] = input[:len(traj)]

    np.savetxt('./data/optimized_traj' +str(j)+'.txt',history, delimiter=",")
plt.show()

