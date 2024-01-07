
import numpy as np
import matplotlib.pyplot as plt


def interpolate(traj, min_dis):
    new_list=[]
    p=0
    for i in range(len(traj)):
        j = (i+1)%len(traj)
        dis_x= traj[j,0]-traj[i,0]
        dis_y= traj[j,1]-traj[i,1]
        dis=np.sqrt(dis_x**2+dis_y**2)
        if dis >min_dis:
            for k in range (int(dis//min_dis)):
                new_point=[traj[i,0]+dis_x*(k+1)/(dis//min_dis+1),traj[i,1]+dis_y*(k+1)/(dis//min_dis+1)]
                new_index = i+k+p+1
                new_list.append([new_point,new_index])
                # new_list.append([[wall[i,0]+dis_x*(k+1)/(dis//min_dis+1),wall[i,1]+dis_y*(k+1)/(dis//min_dis+1)],i+k+p+1])
            p+=int(dis//min_dis)

    refined_traj=insert_new_points(traj,new_list)

    return refined_traj

def insert_new_points(array , new_list):
    new_value = np.zeros(array.shape[1])
    for i in range(len(new_list)):
        new_value[:2] = new_list[i][0]
        if array.shape[1] > 2:
            new_value[2:] = array[new_list[i][1]-1,2:]
        array = np.insert(array,new_list[i][1],new_value ,axis=0)
    return array

def calc_curv(traj):
    traj=np.array(traj)
    # print(len(traj[:,0]))
    dx = np.gradient(traj[:,0])
    dy = np.gradient(traj[:,1])    
    # print(len(dx))
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    # print(len(d2x))
    curvature = (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
    # print(curvature[0])

    return curvature


def calc_bd_width(traj, wall ):
    x=traj[:,0]
    y=traj[:,1]

    bd = []

    # plt.figure()
    # plt.plot(traj[:,0], traj[:,1], '*k')
    # plt.plot(wall[:,0], wall[:,1], '*', color='grey')

    for j in range(len(x)):
        ey = 1000.0
        id = -1
        
        for i in range(len(wall)):
            distance = np.sqrt((x[j]-wall[i,0])**2 + (y[j]-wall[i,1])**2)
            if (distance<ey):
                ey=distance
                id = i 
        
        # plt.plot([x[j], wall[id,0]], [y[j], wall[id,1]], 'r')
        # plt.pause(0.001)
        # plt.draw()

        bd.append(ey)
    return np.array(bd)