#!/usr/bin/env python3

import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
#from tf.transformations import euler_from_quaternion
import sys

sys.path.append('..')

import rospy
import sensor_msgs.msg as sensor_msgs
#from geometry_msgs.msg import Point32
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import LaserScan


rospy.init_node('depth_to_laserscan')

# Create a publisher for the middle row values
#laserscan_pub = rospy.Publisher('/my_laserscan', sensor_msgs.LaserScan, queue_size=1)
#pub_servo = rospy.Publisher("servo", UInt16  , queue_size=10)
pub_midas = rospy.Publisher("midas_depth", Float64, queue_size=10) #Float64
#pub = rospy.Publisher('float_list_topic', Float64MultiArray, queue_size=10)
rate = rospy.Rate(128)

# Q matrix - Camera parameters - Can also be found using stereoRectify
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)

# Load a MiDas model for depth estimation
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

############## Functions ###################

def publish_list(data):
    msg = Float64MultiArray()
    msg.data = data
    pub.publish(msg)
    rospy.loginfo(f"Published float list: {data}")

def calculate_angle(x, y):
    # Calculate angle in degrees from the center bottom
    angle_rad = math.atan2(y - 640, x - 480 // 2)
    angle_deg = math.degrees(angle_rad)
    
    # Ensure the angle is between 0 and 180
    if angle_deg < 0:
        angle_deg += 180
    
    return angle_deg

def depth_array_to_laserscan(depth_array, image_width, frame_id, range_min=0.1, range_max=1000.0, focal_length=4.0):
    """
    Converts a depth array extracted from an image's middle row into a LaserScan message.

    Args:
        depth_array (list): A list of depth values (in meters) corresponding to the middle row of the image.
        image_width (int): Width of the image in pixels.
        frame_id (str): The frame ID of the LaserScan message.
        range_min (float, optional): Minimum range value in meters. Defaults to 0.1.
        range_max (float, optional): Maximum range value in meters. Defaults to 5.0.

    Returns:
        sensor_msgs.LaserScan: The converted LaserScan message.
    """

    import numpy as np
    time_begin = rospy.Time.now()
    while not rospy.is_shutdown():
        i = 0
        time_end = rospy.Time.now()
        for i in range(180):
            pub_servo.publish(i)
            time.sleep(0.1)

            # Calculate horizontal field of view (FOV) based on image dimensions
            # Assuming the depth array represents the middle row (y = image_width // 2)
            #calculate_angle(, )
            horizontal_fov = np.arctan((image_width / 2) / focal_length) * 2  # Adjust for focal length if known

            # Calculate angular increment per pixel
            angle_increment = horizontal_fov / image_width

            # Create a LaserScan message
            d=time_end-time_begin
            laser_scan = sensor_msgs.LaserScan()
            laser_scan.header.seq=d
            laser_scan.header.stamp = time_end
            laser_scan.header.frame_id = frame_id

            # Set scan parameters (assuming FOV covers the entire image width)
            laser_scan.angle_min = -horizontal_fov / 2
            laser_scan.angle_max = horizontal_fov / 2
            laser_scan.angle_increment = angle_increment
            laser_scan.range_min = range_min
            laser_scan.range_max = range_max

            # Fill scan data with depth values, handling potential invalid measurements
            laser_scan.ranges = []
            for depth in depth_array:
	            if depth < range_min:
	                laser_scan.ranges.append(range_min)
	            elif depth > range_max:
	                laser_scan.ranges.append(range_max)
	            else:
	                laser_scan.ranges.append(depth)
            #laser_scan.ranges.append(depth_array[240])
            return laser_scan
    if i==180:
        pub_servo.publish(0)
        #b=b  
    r.sleep()

def calcActualDist(distRow, point2):
    actDistList = []
    horizontal_fov = np.arctan((640 / 2) / 4.0) * 2  # Adjust for focal length if known
    #degree_fov = horizontal_fov * (180/np.pi)
    # Calculate angular increment per pixel
    print("Horizontal FOV in deg: ", horizontal_fov)
    angle_increment = horizontal_fov / 640
    print("Angle Increment", angle_increment)
    angle_increment1 = angle_increment
    i=0
    count = 0
    for point1 in distRow:
        if count % 5 ==0:
            angle_increment1 = angle_increment1 + angle_increment
            x=i
            y=240
            i=i+1
            #ang= calculate_angle(x, y)
            
            #print("Angle inc: ", angle_increment1)
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            # Calculate Euclidean distance
            distanceRow = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            distanceRow = distanceRow/1000
            
            actDistRow = (distanceRow*1.66)+10
            pub_midas.publish(actDistRow)
            #pub_servo.publish(angle_increment1)
            #rate.sleep()
            actDistList.append(actDistRow)
        count +=1

    #pub_midas.publish(actDistList[240])
    #publish_list(actDistList)
    print(actDistList[64])
    rate.sleep()
    return actDistList
        
        
# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply input transforms
    input_batch = transform(img).to(device)
    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    #Get rid of points with value 0 (i.e no depth)
    mask_map = depth_map > 0

    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
    start_point = (0, 400)
    end_point = (640, 400)
    
    middle_row_index = 240 * 640 #output_points.size // 2 + 240 * 640
    middle_row_values = output_points[middle_row_index:middle_row_index + 640]
    val = output_points[425*640+320]
    #print(middle_row_values)
    actDistList = calcActualDist (middle_row_values, val)
    #print(actDistList)
    # ROS parameters (adjust as needed)
    image_width = 640
    image_height = 480
    # ROS parameters (adjust as needed)
    frame_id = 'camera_link'
    range_min = 0.1
    range_max = 1000.0
    # Assuming focal length is unknown (replace if known)
    focal_length = 4.0
    
    ########### Activate laser scan #########
    
    #laser_scan = depth_array_to_laserscan(actDistList, image_width, frame_id, range_min, range_max, focal_length)
    #print(laser_scan)
    #laserscan_pub.publish(laser_scan)
    rate.sleep()
    
    #Row major
    #point1 = output_points[240*640+320]
    #point2 = output_points[425*640+320]
    #x1, y1, z1 = point1
    #x2, y2, z2 = point2
    
    # Calculate Euclidean distance
    #distanceRow = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    #distanceRow = distanceRow/1000
    
    #print("\nThe Euclidean distance is ", distanceRow)
    
    #actDistRow = (distanceRow*1.66)+10
    
    #print("\nThe Actual distance is ", actDistRow)
    end = time.time()
    totalTime = end - start
    print("Total time taken is ", float(totalTime))
    fps = 1 / totalTime
    print("FPS: ", fps)
    cv2.line(img, start_point, end_point, (255, 0, 0), 2)
    cv2.circle(img, (320, 240), 10, (0,0,255), -1)
    cv2.circle(img, (320, 425), 10, (0,0,255), -1)


    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
