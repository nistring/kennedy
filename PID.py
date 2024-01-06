import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from collections import deque
import time

from dynamics import *

@dataclass
class PIDParams():
    dt: float = field(default=0.1)
    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

    noise: bool = field(default=False)
    noise_max: float = field(default=None)
    noise_min: float = field(default=None)

    periodic_disturbance: bool = field(default=False)
    disturbance_amplitude: float = field(default=None)
    disturbance_period: float = field(default=None)

    def default_speed_params(self):
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0
        self.u_min = -5
        self.u_max = 5
        self.du_min = -10 * self.dt
        self.du_max =  10 * self.dt
        self.noise = False
        return

    def default_steer_params(self):
        self.Kp = 1
        self.Ki = 0.0005 / self.dt
        self.Kd = 0
        self.u_min = -0.5
        self.u_max = 0.5
        self.du_min = -4 * self.dt
        self.du_max = 4 * self.dt
        self.noise = False
        return


class PIDLaneFollower():

    def __init__(self, v_ref: float, x_ref: float, dt: float,
                steer_pid_params: PIDParams = None,
                speed_pid_params: PIDParams = None):
        if steer_pid_params is None:
            steer_pid_params = PIDParams()
            steer_pid_params.dt = dt
            steer_pid_params.default_steer_params()
        if speed_pid_params is None:
            speed_pid_params = PIDParams()
            speed_pid_params.dt = dt
            speed_pid_params.default_speed_params()  # these may use dt so it is updated first

        self.dt = dt
        steer_pid_params.dt = dt
        speed_pid_params.dt = dt

        self.steer_pid = PID(steer_pid_params)
        self.speed_pid = PID(speed_pid_params)

        self.v_ref = 2
        self.x_ref = x_ref
        self.speed_pid.initialize(self.v_ref)
        self.steer_pid.initialize(0)

        self.requires_env_state = False
        return

    def initialize(self, **args):
        return

    def solve(self, **args):
        raise NotImplementedError('PID Lane follower does not implement a solver of its own')
        return

    def step(self, vehicle_state: VehicleState, env_state = None):
        v = vehicle_state.v

        vehicle_state.a, _ = self.speed_pid.solve(v)
        # Weighting factor: alpha*x_trans + beta*psi_diff
        alpha = 5.0
        beta = 1.0
        vehicle_state.delta, _ = self.steer_pid.solve(alpha*(vehicle_state.ey - self.x_ref) + beta*vehicle_state.epsi)
        return

class PID():
    def __init__(self, params: PIDParams = PIDParams()):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

        # Add random noise
        self.noise          = params.noise
        self.noise_min      = params.noise_min
        self.noise_max      = params.noise_max

        # Add periodic disturbance
        self.periodic_disturbance = params.periodic_disturbance
        self.disturbance_amplitude = params.disturbance_amplitude
        self.disturbance_period = params.disturbance_period

        self.x_ref          = 0
        self.u_ref          = 0

        self.e              = 0             # error
        self.de             = 0             # finite time error difference
        self.ei             = 0             # accumulated error

        self.time_execution = True
        self.t0 = None

        self.initialized = False

    def initialize(self,
                    x_ref: float = 0,
                    u_ref: float = 0,
                    de: float = 0,
                    ei: float = 0,
                    time_execution: bool = False):
        self.de = de
        self.ei = ei

        self.x_ref = x_ref         # reference point
        self.u_ref = u_ref         # control signal offset

        self.time_execution = time_execution
        self.t0 = time.time()
        self.u_prev = None
        self.initialized = True

    def solve(self, x: float,
                u_prev: float = None) -> Tuple[float, dict]:
        if not self.initialized:
            raise(RuntimeError('PID controller is not initialized, run PID.initialize() before calling PID.solve()'))

        if self.u_prev is None and u_prev is None: u_prev = 0
        elif u_prev is None: u_prev = self.u_prev

        if self.time_execution:
            t_s = time.time()

        info = {'success' : True}

        # Compute error terms
        e_t = x - self.x_ref
        de_t = (e_t - self.e)/self.dt
        ei_t = self.ei + e_t*self.dt

        # Anti-windup
        if ei_t > self.int_e_max:
            ei_t = self.int_e_max
        elif ei_t < self.int_e_min:
            ei_t = self.int_e_min

        # Compute control action terms
        P_val  = self.Kp * e_t
        I_val  = self.Ki * ei_t
        D_val  = self.Kd * de_t

        u = -(P_val + I_val + D_val) + self.u_ref
        if self.noise:
            w = np.random.uniform(low=self.noise_min, high=self.noise_max)
            u += w
        if self.periodic_disturbance:
            t = time.time() - self.t0
            w = self.disturbance_amplitude*np.sin(2*np.pi*t/self.disturbance_period)
            u += w

        # Compute change in control action from previous timestep
        du = u - u_prev

        # Saturate change in control action
        if self.du_max is not None:
            du = self._saturate_rel_high(du)
        if self.du_min is not None:
            du = self._saturate_rel_low(du)

        u = du + u_prev

        # Saturate absolute control action
        if self.u_max is not None:
            u = self._saturate_abs_high(u)
        if self.u_min is not None:
            u = self._saturate_abs_low(u)

        # Update error terms
        self.e  = e_t
        self.de = de_t
        self.ei = ei_t

        if self.time_execution:
            info['solve_time'] = time.time() - t_s

        self.u_prev = u
        return u.__float__(), info

    def set_x_ref(self, x: float, x_ref: float):
        self.x_ref = x_ref
        # reset error integrator
        self.ei = 0
        # reset error, otherwise de/dt will skyrocket
        self.e = x - x_ref

    def set_u_ref(self, u_ref: float):
        self.u_ref = u_ref

    def clear_errors(self):
        self.ei = 0
        self.de = 0

    def set_params(self, params:  PIDParams):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

    def get_refs(self) -> Tuple[float, float]:
        return (self.x_ref, self.u_ref)

    def get_errors(self) -> Tuple[float, float, float]:
        return (self.e, self.de, self.ei)

    def _saturate_abs_high(self, u: float) -> float:
        return np.minimum(u, self.u_max)

    def _saturate_abs_low(self, u: float) -> float:
        return np.maximum(u, self.u_min)

    def _saturate_rel_high(self, du: float) -> float:
        return np.minimum(du, self.du_max)

    def _saturate_rel_low(self, du: float) -> float:
        return np.maximum(du, self.du_min)


def run_pid_warmstart(track, model, t=0, approx=True):

    #x = x,y, psi, s, ey, epsi,v, a, delta
    ego_sim_state = model.vehicle_state.initialize()
    s0 = model.vehicle_state.initialize()
    N = model.N
    
    state_history_ego = deque([], N)
    input_history_ego = deque([], N)

    input_ego = VehicleActuation()

    x_ref = ego_sim_state.ey

    ds =  model.ds
    pid_steer_params = PIDParams()
    pid_steer_params.ds = ds
    pid_steer_params.default_steer_params()
    pid_steer_params.Kp = 1
    pid_speed_params = PIDParams()
    pid_speed_params.ds = ds
    pid_speed_params.default_speed_params()

    pid_controller = PIDLaneFollower(ego_sim_state.v, x_ref, ds, pid_steer_params, pid_speed_params)
    egost_list = [model.vehicle_state.initialize()]

    n_iter = 0.0
    s_prev = 0.0
    while n_iter < N:
        pid_controller.step(ego_sim_state)
        
        model.step(ego_sim_state)

        
        input_ego.t = t
        input_ego.a = ego_sim_state.a
        input_ego.delta = ego_sim_state.delta
        q, _ = model.state2qu(ego_sim_state)
        u = model.input2u(input_ego)

        state_history_ego.append(q)
        input_history_ego.append(u)

        egost_list.append(ego_sim_state.initialize())

        n_iter += 1
        t += ds
        s_prev = q[2]
        # print(n_iter)
        

    compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))

    return compose_history(state_history_ego, input_history_ego), s0, egost_list,


