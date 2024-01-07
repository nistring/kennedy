#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import itertools,operator

#ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

prev_steering_angle = 0
t_prev = 0.0
prev_range = []

view_angel = 90 #0-135
view_idx = int(1077/135*view_angel)

Kp = 1
Ki = 1
Kd = 1

class reactive_follow_gap:
    def __init__(self):
        #Topics & Subscriptions,Publishers
        self.lidar_sub = rospy.Subscriber('/scan',LaserScan, self.lidar_callback)
        self.drive_pub = rospy.Publisher('/vesc/ackermann_cmd',AckermannDriveStamped,queue_size=5)

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop


        ###########################3
        
        # Return the start index & end index of the max gap in free_space_ranges

        # print(free_space_ranges)
        # # print(len(free_space_ranges))
        
        # start_i = end_i = None
        # max_len = 0
        # gap_start_i = None

        # for i in range(len(free_space_ranges)):
        #     if free_space_ranges[i] > 0.2 and gap_start_i is None:
        #         gap_start_i = i

        #         # print(i)

        #     elif free_space_ranges[i] <= 0.2 and gap_start_i is not None:
        #         gap_len = i - gap_start_i
        #         if gap_len > max_len:
        #             max_len = gap_len
        #             start_i = gap_start_i
        #             end_i = i
        #         gap_start_i = None

        #         # print(2)

        #return start_i, end_i
    
    def find_best_point(self, start_i, end_i, ranges):
        # Start_i & end_i are start and end indices of max-gap range, respectively
        # Return index of best point in ranges
        # Naive: Choose the furthest point within ranges and go there
        
        best_i = None
        best_range = 0
        
        for i in range(start_i, end_i):
            if ranges[i] > best_range:
                best_range = ranges[i]
                best_i = i
                
        return best_i

    def lidar_callback(self, data):

        print("@@@@@@@@@@@@@@@@@@222")
        
        # Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message

        #TO DO:  
        # Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        #       1. Get LiDAR message
        #       2. Find closest point to LiDAR
        #       3. Eliminate all points inside 'bubble' (set them to zero) 
        #       4. Fine the max gap -> "def find_max_gap(self, free_space_ranges):"
        #       4. Find the best point -> "def find_best_point(self, start_i, end_i, ranges):"
        #       5. Publish Drive message

        # Get LiDAR message

        
        ranges = np.array(data.ranges)[1077-view_idx:1078+view_idx]
        # angles = np.arange(data.angle_min, data.angle_max+data.angle_increment, data.angle_increment)
        # angles = np.arange()

        # print(angles)

        # Find closest point to LiDAR
        closest_range = np.min(ranges)
        closest_i = np.argmin(ranges)

        # Eliminate all points inside 'bubble' (set them to zero) 
        bubble_radius = 300 # index //meters
        # bubble_indices = np.logical_and(
        #     ranges > 0,
        #     ranges < bubble_radius
        # )

        for i in range(bubble_radius):
            j = int(closest_i-bubble_radius/2+i)

            if j>=0 and j<len(ranges):
                ranges[j]=0
        # ranges[closest_i-bubble_radius/2:closest_i+bubble_radius/2] = 0

        # Find max gap and best point within gap
        start_i, end_i = self.find_max_gap(ranges)

        if start_i == None and end_i == None:
            start_i = 0

        best_i = self.find_best_point(start_i, end_i, ranges)

        print(best_i)
        print("$$$$$$$$$$")

        # Calculate steering angle
        # steering_angle = angles[best_i]


        
        steering_angle = (best_i-view_idx+1)/view_idx*90/180*np.pi

        print("steer: {}".format(steering_angle))

        # Publish Drive message
        # global prev_steering_angle, t_prev, prev_range
        if closest_range < 0.2:
            speed = 0.5
        else:
            speed = 0.7

        # print(speed)
        # print("#############################")

        
        # steering_angle = prev_steering_angle + Kp*()


        

        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = speed
        self.drive_pub.publish(self.drive_msg)



def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

