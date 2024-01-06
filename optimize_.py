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

    #Qs: float = field(default=20.0)
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
    e_psi_max: float = field(default=np.pi)
    e_psi_min: float = field(default=-np.pi)
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
        self.state_ub = np.array([self.ey_max,
                                  self.e_psi_max,
                                  self.v_max])

        self.state_lb = np.array([self.ey_min,
                                  self.e_psi_min,
                                  self.v_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])

        return    

opt_params = ModelParams(
    solver_dir='/home/hmcl/Download/Track_Optimization/solver',
    optlevel=2,
    N= 70,

    #Qs= 1, 
    Qey= 1,
    Qv = 1,
    Qepsi= 5, #vy

    R_a= 0.5,
    R_delta= 0.5,

    
    v_max= 8.0,
    v_min= 0.0,
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

        # Dynamics Params
        self.dynamics = dynamics
        self.lencar = dynamics.L  

        self.ds = dynamics.ds
        self.Nx = dynamics.n_q # number of states
        self.Nu = dynamics.n_u # number of inputs
        self.N = dynamics.N


        # optimization params
        self.control_params = control_params
        #self.Ts = self.dt
        self.Ss = self.track_length / self.N
        self.ds = self.track_length / self.N
        
        # self.T = ca.SX.sym('T')
        # self.dt = self.T/self.N

        #self.Qs = ca.SX(control_params.Qs)
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

        # self.X = ca.SX.sym('X', (self.N + 1)*self.Nx)
        self.X = ca.SX.sym('X', self.N*self.Nx) #  s 제외
        self.U = ca.SX.sym('U', self.N*self.Nu)
        

        self.u_prev = np.zeros(self.Nu)
        self.x_pred = np.zeros((self.N, self.Nx))
        self.u_pred = np.zeros((self.N, self.Nu))
        self.x_ws = None
        self.u_ws = None

        self.cost = 0.0
        self.const = []
        self.lbx = []
        self.ubx = []

    def solve_optimization(self, state):

        x0, _ = self.dynamics.state2qu(state)
        print(x0)
        ## Cost 
        for t in range(self.N):
            estimated_distance = self.X[self.Nx*t+2]*self.ds
            self.cost += estimated_distance * self.Qv
            #self.cost += self.dt
            # self.cost += ca.if_else(self.X[self.Nx*t]>self.track_length, 0, self.dt)
            # self.cost -= self.Qs*(self.X[self.Nx*(t+1)] - self.X[self.Nx*t])
            #self.cost -= (self.X[self.Nx*(t+1)] - self.X[self.Nx*t])*self.Qs
            #self.cost -= (self.X[self.Nx*(t+1)] - self.X[self.Nx*t])*self.Qs*(self.X[self.Nx*(t+1)] - self.X[self.Nx*t])

            # self.cost += self.X[self.Nx*t+1]*self.Qey*self.X[self.Nx*t+1]
            #self.cost += self.X[self.Nx*t+2].T*self.Qepsi*self.X[self.Nx*t+2]


            # if t < self.N-1:
            #     # v_bar = self.X[self.Nx*(t+1)+3] - self.X[self.Nx*t+3] 
            #     u_bar = self.U[self.Nu*(t+1):self.Nu*(t+2)] - self.U[self.Nu*t:self.Nu*(t+1)]
            #     # u_bar = self.U[self.Nu*t:self.Nu*(t+1)]

            #     self.cost += u_bar[0]*self.R_a*u_bar[0]
            #     self.cost += u_bar[1]*self.R_delta*u_bar[1]

        self.const=[]
        ## Constraint for dynamics
        for t in range(self.N):
            current_x = self.X[self.Nx*t:self.Nx*(t+1)]
            current_u = self.U[self.Nu*t:self.Nu*(t+1)]

            integrated = self.dynamics.f_d_rk4(current_x, current_u, self.ds)
            # integrated = self.dynamics.update(current_x, current_u)
            self.const = ca.vertcat(self.const, self.X[self.Nx*(t):self.Nx*(t+1)]-integrated)
        ####
        # initial_s_position = x0[0]
        # final_s_position = self.X[self.Nx*self.N]
        # lap_completion_constraint = ca.fabs(final_s_position-(initial_s_position+self.track_length))-lap_completion_constraint
        # self.const = ca.vertact(self.const, lap_completion_constraint)
        #self.const = ca.vertcat(self.const, self.X[self.Nx*self.N])
        print(self.const.size1())
        ## Constraint for state (upper bound and lower bound)
        self.lbx = [0] 
        self.ubx = [100]
        #self.lbx +=list(x0)
        #self.ubx +=list(x0) # what is it?
        
         
        for t in range(self.N):
            self.lbx += list(self.state_lb)
            self.ubx += list(self.state_ub) 



           
        ##origin
        # self.lbx += list(x0)
        # self.ubx += list(x0) 
        # self.lbx[0] = 0
        # self.ubx[0] = self.track_length         
        #for t in range(self.N):
            # left_bd = self.dynamics.get_left_bd(self.X[self.Nx*t]) #ey 
            # right_bd = self.dynamics.get_right_bd(self.X[self.Nx*t])
            # self.state_ub[1] = left_bd
            # self.state_lb[1] = -right_bd

            # self.ubx +=  list(self.state_ub)
            # self.lbx +=  list(self.state_lb)
        ## origin

        self.ubx +=  list(self.input_ub)*self.N
        self.lbx +=  list(self.input_lb)*self.N

        self.lbg_dyanmics = [0]*(self.Nx*self.N)+[0]
        self.ubg_dyanmics = [0]*(self.Nx*self.N)+[100]

        total_variable = self.X.size1() + self.U.size1()
        print(self.X.size1(), self.X.size2(), self.U.size1(), self.U.size2())
        lbx_length = len(self.lbx)
        ubx_length = len(self.ubx)
        lbg_length = len(self.lbg_dyanmics)
        ubg_length = len(self.ubg_dyanmics)
        print("Total:", total_variable, "lbx,ubx,lbg,ubg:",lbx_length, ubx_length, lbg_length, ubg_length)
        
        ######SOLVE
        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0} #, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        nlp = {'x':ca.vertcat(self.X,self.U), 'f':self.cost, 'g':self.const }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)


        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        #s_init = self.Ss*self.N
        # x_ws = np.zeros((self.N+1, self.Nx))
        # x_ws[0] = x0
        # x_ws[1:] = self.x_ws
        x_ws = np.zeros((self.N, self.Nx))
        x_ws = self.x_ws
        
        u_ws = np.concatenate(self.u_ws)
        #x_init = ca.vertcat(s_init,np.concatenate(x_ws))
        x_init = ca.vertcat(np.concatenate(x_ws), u_ws)

        sol = self.solver(x0=x_init, lbx = self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics,ubg=self.ubg_dyanmics)

        x = sol['x']
        g = sol['g']

        opt_time = x[0]
        print("optimal_time:", x[0])
        print(x)
        idx = 1+(self.N+1)*self.Nx
        for i in range(0,self.N):
            self.u_pred[i, 0] = x[idx+self.Nu*i].__float__()
            self.u_pred[i, 1] = x[idx+self.Nu*i + 1].__float__()

        for i in range(0,self.N):
            self.x_pred[i, 0] = x[1+self.Nx*i].__float__()
            self.x_pred[i, 1] = x[1+self.Nx*i + 1].__float__()
            self.x_pred[i, 2] = x[1+self.Nx*i + 2].__float__()
            #self.x_pred[i, 3] = x[1+self.Nx*i + 3].__float__()

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