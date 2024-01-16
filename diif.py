import numpy as np

# data = np.loadtxt("/home/hmcl/Downloads/kennedy-main/data/optimized_traj.txt", delimiter=",", dtype=float)
data = np.loadtxt("/home/hmcl/Downloads/Track_Optimization/data/optimized_traj.txt", delimiter=",", dtype=float)

x= np.abs(data[:,0])
#print(x)
y= np.abs(data[:,1])
#print(y)
print(x.shape[0])
squared_sum = np.sqrt(x**2+y**2)

for i in range(198):    
    diff_front_and_diff_pre = np.abs(squared_sum[i+1]-squared_sum[i])
    print(diff_front_and_diff_pre)
    