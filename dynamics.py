from dataclasses import dataclass, field
import numpy as np

import casadi as ca
from utils import *
import copy
import matplotlib.pyplot as plt

class VehicleState():
    def __init__(self):
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0 
        self.psi = 0.0 
        self.s = 0.0 
        self.ey = 0.0 
        self.epsi = 0.0 
        self.v = 0.0 
        self.a = 0.0
        self.delta = 0.0 

    def initialize(self):
        return copy.deepcopy(self) #shallow copy


@dataclass
class VehicleActuation(PythonMsg):
    t: float = field(default=0)

    a: float = field(default=0)
    delta: float = field(default=0)

    def __str__(self):
        return 't:{self.t}, a:{self.a}, delta:{self.delta}'.format(self=self)


class KinematicBicycle():
    def __init__(self, track, dt=0.1, N=70):

        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)
        self.dt = dt
        self.t0 = 0.0
        self.M = 1 # RK4 integration steps
        self.h = self.dt/self.M
        self.N = N

 
        self.vehicle_state = VehicleState()
        self.vehicle_state.x = track.x[0]
        self.vehicle_state.y = track.y[0]
        self.vehicle_state.psi = track.psi[0]

        self.track.global_to_local_typed(self.vehicle_state)

        ''' Define the bicycle dynamics
            x = [s, ey, epsi, v]
            u = [a, delta] longitudinal acceleration, steering angle'''

        # Number of states and input
        self.n_q = 4 #s,ey,epsi,v
        self.n_u = 2 #a,delta

        # Model Prameter
        self.L = 0.55 #Length
        self.L_f = 0.17 #wheel_dist_front
        self.L_r = 0.17 #wheel_dist_rear

        self.sym_s = ca.SX.sym('s')
        self.sym_ey = ca.SX.sym('ey')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_v = ca.SX.sym('v')

        # Track parameters
        self.track_length = ca.SX.sym('track_length')
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.sym_c = self.get_curvature(self.sym_s)

        self.get_left_bd = self.track.get_left_bd_casadi_fn()
        self.sym_left_bd = self.get_left_bd(self.sym_s)
        self.get_right_bd = self.track.get_right_bd_casadi_fn()
        self.sym_right_bd = self.get_right_bd(self.sym_s)

        self.sym_u_a = ca.SX.sym('a')
        self.sym_u_s = ca.SX.sym('delta')
        
        self.initialize_model()
    
    
    def initialize_model(self):
       
        # time derivatives
        self.sym_ds = self.sym_v * ca.cos(self.sym_epsi) / (1 - self.sym_ey * self.sym_c)
        self.sym_dey = self.sym_v * ca.sin(self.sym_epsi)
        self.sym_depsi = self.sym_v*ca.tan(self.sym_u_s)/(self.L) - self.sym_ds*self.sym_c
        self.sym_dv = self.sym_u_a - self.sym_v*self.sym_c
        
        # self.sym_ds = self.sym_v * ca.cos(self.sym_epsi)
        # self.sym_dey = self.sym_v * ca.sin(self.sym_epsi)
        # self.sym_depsi = self.sym_v*ca.tan(self.sym_u_s)/(self.L_f + self.L_r)
        # self.sym_dv = self.sym_u_a - self.sym_v*self.sym_c*self.sym_depsi
       

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_s, self.sym_ey, self.sym_epsi, self.sym_v)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_ds, self.sym_dey, self.sym_depsi, self.sym_dv)

        self.precompute_model()


    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.SX with elements of state vector q
        self.sym_u:  ca.SX with elements of control vector u
        self.sym_dq: ca.SX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        dyn_inputs = [self.sym_q, self.sym_u]

        # Continuous time dynamics function
        self.fc = ca.Function('fc', dyn_inputs, [self.sym_dq], self.options('fc'))

        # First derivatives
        self.sym_Ac = ca.jacobian(self.sym_dq, self.sym_q)
        self.sym_Bc = ca.jacobian(self.sym_dq, self.sym_u)
        self.sym_Cc = self.sym_dq

        self.fA = ca.Function('fA', dyn_inputs, [self.sym_Ac], self.options('fA'))
        self.fB = ca.Function('fB', dyn_inputs, [self.sym_Bc], self.options('fB'))
        self.fC = ca.Function('fC', dyn_inputs, [self.sym_Cc], self.options('fC'))

        # Discretization with euler
        sym_q_kp1 = self.sym_q + self.sym_ds * self.fc(*dyn_inputs) / (self.sym_v + 1e-6)
        # sym_q_kp1 = self.f_d_rk4(*dyn_inputs, self.dt)
        
        # Discrete time dynamics function
        self.fd = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        # First derivatives
        self.sym_Ad = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd = sym_q_kp1

        self.fAd = ca.Function('fAd', dyn_inputs, [self.sym_Ad], self.options('fAd'))
        self.fBd = ca.Function('fBd', dyn_inputs, [self.sym_Bd], self.options('fBd'))
        self.fCd = ca.Function('fCd', dyn_inputs, [self.sym_Cd], self.options('fCd'))

        # Second derivatives
        self.sym_Ed = [ca.hessian(sym_q_kp1[i], self.sym_q)[0] for i in range(self.n_q)]
        self.sym_Fd = [ca.hessian(sym_q_kp1[i], self.sym_u)[0] for i in range(self.n_q)]
        self.sym_Gd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_q) for i in range(self.n_q)]
        
        self.fEd = ca.Function('fEd', dyn_inputs, self.sym_Ed, self.options('fEd'))
        self.fFd = ca.Function('fFd', dyn_inputs, self.sym_Fd, self.options('fFd'))
        self.fGd = ca.Function('fGd', dyn_inputs, self.sym_Gd, self.options('fGd'))

    # def f_d_rk4(self, x, u):
    #     '''
    #     Discrete nonlinear dynamics (RK4 approx.)
    #     '''
    #     x_p = x
    #     for i in range(self.M):
    #         a1 = self.fc(x_p, u)
    #         a2 = self.fc(x_p + (self.h / 2) * a1, u)
    #         a3 = self.fc(x_p + (self.h / 2) * a2, u)
    #         a4 = self.fc(x_p + self.h * a3, u)
    #         x_p += self.h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
    #     return x_p
    
    def f_d_rk4(self, x, u, dt):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        h = dt/self.M
        for i in range(self.M):
            a1 = self.fc(x_p, u)
            a2 = self.fc(x_p + (h / 2) * a1, u)
            a3 = self.fc(x_p + (h / 2) * a2, u)
            a4 = self.fc(x_p + h * a3, u)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
            
        return x_p



    def update(self, q, u):
        curv = self.get_curvature(q[0])
        s_new = q[0] + q[3]*np.cos(q[2])*self.dt/(1-q[1]*curv)
        ey_new = q[1] + q[3]*np.sin(q[2])*self.dt
        epsi_new = q[2] + q[3]*ca.tan(u[1])/(self.L_f + self.L_r) - q[3]*np.cos(q[2])*curv*q[3]/(1-q[1]*curv)
        v_new = q[3] + u[0]*self.dt

        return np.array([s_new, ey_new, epsi_new, v_new])


    def step(self, state):
         
        q, u = self.state2qu(state)  

        #Update time
        t = state.t - self.t0
        tf = state.t + self.t0
        state.t = tf + self.t0

        # q_new = self.update(q,u)
        q_new = np.array(self.f_d_rk4(q, u, self.dt))[:,0]

        self.qu2state(state, q_new, u)  
        self.track.local_to_global_typed(state)


    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.s = q[0]
            state.ey = q[1]
            state.epsi = q[2]
            state.v = q[3]

        if u is not None:
            state.a = u[0]
            state.delta = u[1]
        return
    

    def input2u(self, input):
        u = np.array([input.a, input.delta])
        return u


    def state2qu(self, state):
        q = np.array([state.s, state.ey, state.epsi, state.v])
        u = np.array([state.a, state.delta])
        return q, u
