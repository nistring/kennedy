
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import casadi as ca

from track_utils import *

class Track_irl():
    def __init__(self, center, data):
        
        # self.track = interpolate(center, 0.1)
        # self.data = interpolate(data, 0.1)
        self.track = center
        self.data = data

        self.get_track_info() # compute s, psi, length
        data_lc_coord = self.get_data_info()

        self.data_save(data_lc_coord)

    def data_save(self,data_lc_coord):
        
        data = np.zeros((len(self.data),6)) #s,ey,epsi,v,a,steering

        data[:,0] = data_lc_coord[:,0]
        data[:,1] = data_lc_coord[:,1]
        data[:,2] = data_lc_coord[:,2]
        data[:,3:] = self.data[:,3:]

        self.n_data = data

    def get_track_info(self):
        dx = np.gradient(self.track[:,0])
        dy = np.gradient(self.track[:,1])
        ds = np.sqrt(dx**2 + dy**2)
        
        self.track_psi = np.arctan2(dy,dx)
        self.track_s = np.cumsum(ds)
        self.track_length = np.sum(ds)
        self.track_curv = calc_curv(self.track)

    def get_data_info(self):
        
        data_lc_coord = []
        for idx in range(len(self.data)):
            x = self.data[idx,0]
            y = self.data[idx,1]
            psi = self.data[idx,2]

            pos_cur = np.array([x, y])
            cl_coord = None
            
            distance = 100
            current_idx = -1
            for i in range(len(self.track_s)-1):
                x_s = self.track[i,0]
                y_s = self.track[i,1]
                pos_s = np.array([x_s, y_s])
                # print(la.norm(pos_s - pos_cur))
                if distance >= la.norm(pos_s - pos_cur):
                    current_idx = i
                    distance = la.norm(pos_s - pos_cur)
                    # print(distance)
            e_y = -np.sin(psi) * (x - self.track[current_idx,0]) + np.cos(psi) * (y - self.track[current_idx,1])
            if np.isnan(e_y):
                e_y = 0.0
            e_psi = psi - self.track_psi[current_idx]#np.arctan2(self.track_curv[current_idx] * e_y, 1)
            s = self.track_s[current_idx]
            cl_coord = [s, e_y, e_psi]
            data_lc_coord.append(cl_coord) 

        return np.array(data_lc_coord)

def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta

# if __name__ == "__main__": 
    # Read history data 

    