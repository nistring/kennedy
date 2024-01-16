import sys, os, pathlib, shutil, pickle

import casadi as ca
import numpy as np
from dataclasses import dataclass, field, fields

from dynamics import VehicleActuation
import matplotlib.pyplot as plt


@dataclass
class ModelParams():
    n: int = field(default=3) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=50)

    Qs: float = field(default=20.0)
    Qey: float = field(default=80.0)
    Qv: float = field(default=80.0)
    Qepsi: float = field(default=80.0)
   
    R_a: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    # named constraint
    #s_max: float = field(default=np.inf)
    #s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=np.pi/2.1)
    e_psi_min: float = field(default=-np.pi/2.1)
    v_max: float = field(default=np.inf)
    v_min: float = field(default=-np.inf)

    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_steer_rate_max: float     = field(default = 10)
    u_steer_rate_min: float     = field(default = -10)


    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        # TODO Temporary fix
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([
                                  self.ey_max,
                                  self.e_psi_max,
                                  self.v_max])

        self.state_lb = np.array([
                                  self.ey_min,
                                  self.e_psi_min,
                                  self.v_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])

        return    

opt_params = ModelParams(
    solver_dir='/home/im/Desktop/IterativeTrackOptimization/solver',
    optlevel=2,
    N= 70,

    Qs= 1, 
    Qey= 1,
    Qv = 1,
    Qepsi= 5, #vy

    R_a= 0.5,
    R_delta= 0.5,

    
    v_max= 5.5,
    v_min= 0.5,
    u_a_min= -3,
    u_a_max= 3,
    
    u_steer_max=0.5,
    u_steer_min=-0.5,

    u_a_rate_max=3,
    u_a_rate_min=-3,
    u_steer_rate_max=5,
    u_steer_rate_min=-5
)


class Optimize():
    def __init__(self, dynamics, track, control_params):
        
        '''
        Define Parameters
        '''

        # Track Params
        self.track = track
        self.track_length = track.track_length
        self.N = dynamics.N

        curv_info = track.get_curvature_steps(self.N)
        self.s_values, self.curv_values = zip(*curv_info)
        # Dynamics Params
        self.dynamics = dynamics
        self.lencar = dynamics.L  

        self.Nx = dynamics.n_q # number of states
        self.Nu = dynamics.n_u # number of inputs
        

        # optimization params
        self.control_params = control_params

        self.Qs = ca.SX(control_params.Qs)
        self.Qey = ca.SX(control_params.Qey)
        self.Qv = ca.SX(control_params.Qv)
        self.Qepsi = ca.SX(control_params.Qepsi)

        self.R_a = control_params.R_a
        self.R_delta = control_params.R_delta

        # Input Box Constraints
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.state_ub[0] = track.track_length*2
        self.state_lb[0] = -track.track_length
        self.state_ub[1] = 0.5#min(max(self.track.left_bd),max(self.track.right_bd))
        self.state_lb[1] =-0.5#1*min(max(self.track.left_bd),max(self.track.right_bd))

        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        
        '''
        Define Variables for optimization
        '''

        self.X = ca.SX.sym('X', (self.N + 1)*self.Nx)
        self.U = ca.SX.sym('U', self.N*self.Nu)
        

        self.u_prev = np.zeros(self.Nu)
        self.x_pred = np.zeros((self.N, self.Nx))
        self.u_pred = np.zeros((self.N, self.Nu))
        self.x_ws = None
        self.u_ws = None

        self.cost = 0.0
        self.const = []

    def solve_optimization(self, state):
        
        x0, _ = self.dynamics.state2qu(state)
        print(x0)
        ## Cost
        # https://ieeexplore.ieee.org/document/9340640 (7)
        ds = self.track_length / (self.N - 1)
        print("tracklength,ds", self.track_length,ds)
        v = self.X[3::self.Nx]
        v = (v[1:] + v[:-1]) / 2
        for i in range(v.shape[0]):
            self.cost += (ds / (v[i] + 1e-12)) * self.Qs
        
        # for t in range(self.N):
        #     self.cost += self.X[self.Nx*t+1]*self.Qey*self.X[self.Nx*t+1]
        #     self.cost += self.X[self.Nx*t+2]*self.Qepsi*self.X[self.Nx*t+2]

        ## Constraint for dynamics
        for t in range(self.N):
            
            current_x = self.X[self.Nx*t:self.Nx*(t+1)]
            current_u = self.U[self.Nu*t:self.Nu*(t+1)]
            next_x = self.X[self.Nx*(t+1):self.Nx*(t+2)]
            
            # x(n+1) = x(n) + f(x, u)/v * ds
            self.const = ca.vertcat(self.const, next_x - self.dynamics.f_d_rk4(current_x, current_u,ds))
            # x(n, 0) = s(n)
            self.const = ca.vertcat(self.const, self.track_length * t / self.N - current_x[0])
        # x(N, 0) = s(N)
        self.const = ca.vertcat(self.const, self.track_length - self.X[self.Nx * self.N])

        ## Constraint for state (upper bound and lower bound)
        self.lbx = list(x0)
        self.ubx = list(x0) 

        self.ubx +=  list(self.state_ub)*self.N
        self.lbx +=  list(self.state_lb)*self.N

        self.ubx +=  list(self.input_ub)*self.N
        self.lbx +=  list(self.input_lb)*self.N

        self.lbg_dyanmics = [0]*((self.Nx+1)*self.N + 1)
        self.ubg_dyanmics = [0]*((self.Nx+1)*self.N + 1)

        ######SOLVE
        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0} #, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        nlp = {'x':ca.vertcat(self.X,self.U), 'f':self.cost, 'g':self.const}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)


        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        x_ws = np.zeros((self.N+1, self.Nx))
        x_ws[0] = x0
        x_ws[1:] = self.x_ws
        x_init = ca.vertcat(np.concatenate(x_ws), np.concatenate(self.u_ws))

        sol = self.solver(x0=x_init, lbx = self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics,ubg=self.ubg_dyanmics)

        x = sol['x']
        print(x)
        g = sol['g']
        print("opt", x[0])
        idx = (self.N+1)*self.Nx
        for i in range(0,self.N):
            self.u_pred[i, 0] = x[idx+self.Nu*i].__float__()
            self.u_pred[i, 1] = x[idx+self.Nu*i + 1].__float__()

        for i in range(0,self.N):
            self.x_pred[i, 0] = x[self.Nx*i].__float__()
            self.x_pred[i, 1] = x[self.Nx*i + 1].__float__()
            self.x_pred[i, 2] = x[self.Nx*i + 2].__float__()
            #self.x_pred[i, 3] = x[self.Nx*i + 3].__float__()

        # for i in range(0,self.N-1):
        #     s_next = x[1+self.Nx*(i+1)].__float__()
        #     s_ = x[1+self.Nx*(i)].__float__()
        #     vel = (s_next -s_)/(opt_time/self.N)
        #     print(vel)
        x = np.zeros((len(self.x_pred),1))
        y = np.zeros((len(self.x_pred),1))
        psi = np.zeros((len(self.x_pred),1))
        
        for i in range(0,len(self.x_pred)):
            x[i], y[i], psi[i] = self.track.local_to_global(np.array([self.x_pred[i,0], self.x_pred[i,1], self.x_pred[i,2]]))
        
        plt.clf()
        plt.plot(self.track.x,self.track.y,'k')
        plt.plot(self.track.inner[:,0],self.track.inner[:,1],'k')
        plt.plot(self.track.outer[:,0],self.track.outer[:,1],'k')
        plt.plot(x, y,'go')
        plt.show()

        states = np.zeros((len(self.x_pred),3))
        states[:,0] = x.squeeze()
        states[:,1] = y.squeeze()
        states[:,2] = psi.squeeze()

        

        return self.x_pred, self.u_pred, states

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.Nx:  # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N, self.Nx)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.Nu:
            raise (RuntimeError(
                'Warm start input sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.Nu)))

        self.x_ws = x_ws
        self.u_ws = u_ws