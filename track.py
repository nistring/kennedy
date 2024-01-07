
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import casadi as ca



# from scipy.optimize import minimize, LinearConstraint
# import forcespro.nlp
# import casadi as ca
# from matplotlib.widgets import Cursor
# from matplotlib.collections import LineCollection

from track_utils import *

class Track():
    def __init__(self, center, inner, outer):
        
        center = interpolate(center, 0.1)
        self.inner = inner
        self.outer = outer
        self.center = center
        
        self.x = center[:,0]
        self.y = center[:,1]
        self.v = center[:,2]
        self.curv = calc_curv(center)

        self.get_track_info() # compute s, psi, length

        fined_inner = interpolate(inner, 0.1)
        fined_outer = interpolate(outer, 0.1)
        self.left_bd = calc_bd_width(center, fined_inner)
        self.right_bd = calc_bd_width(center, fined_outer)

    def get_track_info(self):
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        ds = np.sqrt(dx**2 + dy**2)
        
        self.psi = np.arctan2(dy,dx)
        self.s = np.cumsum(ds)
        self.track_length = np.sum(ds)

    
    def get_curvature_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)

        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        pw_const_curvature = ca.pw_const(sym_s_bar, self.s[1:-1], self.curv[1:])
        return ca.Function('track_curvature', [sym_s], [pw_const_curvature])
    
    def get_left_bd_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)

        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        pw_const_left_bd = ca.pw_const(sym_s_bar, self.s[1:-1], self.left_bd[1:])
        return ca.Function('track_curvature', [sym_s], [pw_const_left_bd])
    
    def get_right_bd_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)

        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.fmod(ca.fmod(sym_s, self.track_length) + self.track_length, self.track_length)
        pw_const_right_bd = ca.pw_const(sym_s_bar, self.s[1:-1], self.right_bd[1:])
        return ca.Function('track_curvature', [sym_s], [pw_const_right_bd])



    
    
    def local_to_global(self, cl_coord):

        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]


        if s <= self.s[0]:
            idx_s = 0
        else:
            idx_s = np.where(s >= self.s)[0][-1]


        x_s = self.x[idx_s]
        y_s = self.y[idx_s]
        psi_s = self.psi[idx_s]
        curv = self.curv[idx_s]
        d = s - self.s[idx_s]


        r = 1 / curv
        dir = np.sign(r)

        # Find coordinates for center of curved segment
        x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
        y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)

        # Angle spanned up to current location along segment
        span_ang = d / np.abs(r)

        # Angle of the tangent vector at the current location
        psi_d = wrap_angle(psi_s + dir * span_ang)

        ang_norm = wrap_angle(psi_s + dir * np.pi / 2)
        ang = -np.sign(ang_norm) * (np.pi - np.abs(ang_norm))

        x = x_c + (np.abs(r) - dir * e_y) * np.cos(ang + dir * span_ang)
        y = y_c + (np.abs(r) - dir * e_y) * np.sin(ang + dir * span_ang)
        psi = wrap_angle(psi_d + e_psi)
        
        return (x, y, psi)

    def local_to_global_typed(self, data):
        cl_coord = (data.s, data.xtran, data.epsi)
        xy_coord = self.local_to_global(cl_coord)
        if xy_coord:
            data.x = xy_coord[0]
            data.y = xy_coord[1]
            data.psi = xy_coord[2]
            return -1
        return 0
    
    def global_to_local_typed(self, data):  # data is vehicleState
        xy_coord = (data.x, data.y, data.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.s = cl_coord[0]
            data.xtran = cl_coord[1]
            data.e_psi = cl_coord[2]
            return 0
        return -1
    
    def global_to_local(self, xy_coord):


        x = xy_coord[0]
        y = xy_coord[1]
        psi = xy_coord[2]

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(len(self.s)-1):
            x_s = self.x[i]
            y_s = self.y[i]
            psi_s = self.psi[i]
            curve_s = self.curv[i]

            x_f = self.x[i+1]
            y_f = self.y[i+1]
            psi_f = self.psi[i+1]
            curve_f = self.curv[i+1]


            l = self.s[i+1]-self.s[i]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.s[i]
                e_y = 0
                e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, e_y, e_psi)
                break
            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.s[i+1]
                e_y = 0
                e_psi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, e_y, e_psi)
                break

            if curve_f == 0:
                # Check if on straight segment
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
                        compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v) * np.sin(ang)
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    width = np.min(inner_width,outer_width)
                    if np.abs(e_y) <= width + self.slack:
                        d = la.norm(v) * np.cos(ang)
                        s = self.s[i] + d
                        e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                # Check if on curved segment
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    width = np.min(inner_width,outer_width)
                    if np.abs(e_y) <= width + self.slack:
                        d = np.abs(cur_ang) * np.abs(r)
                        s = self.s[i] + d
                        e_psi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
                        cl_coord = (s, e_y, e_psi)
                        break
                    else:
                        continue
                else:
                    continue

        return cl_coord

def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle

