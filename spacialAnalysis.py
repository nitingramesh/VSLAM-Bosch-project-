import cv2
import torch
import time
import numpy as np
import rospy
from std_msgs.msg import String

pub = rospy.Publisher('depth', String, queue_size=10)

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
    print("\n\n")
    k =m = r= False
    center_rel = depth_map[320, 240]
    obj_rel = depth_map[330, 500]
    for i in range(0, 400):
    	for j in range(0, 210):
    		if depth_map[i, j] > 0.65:
    			k = True
    			print("Right Val: ", depth_map[i, j])
    			input_depth = depth_map[i, j]
    			
    			break
    	if k==True:
	    		break
    if k==False:
	    for i in range(0, 400):
	    	for j in range(211, 420):
	    		if depth_map[i, j] > 0.65:
	    			r = True
	    			print("Center Val: ", depth_map[i, j])
	    			input_depth = depth_map[i, j]
	    			
	    			break
	    	if r==True:
	    		break
    
    if k==False and r==False:
	    for i in range(0, 400):
	    	for j in range(421, 640):
	    		if depth_map[i, j] > 0.65:
	    			m = True
	    			print("Left Val: ", depth_map[i, j])
	    			input_depth = depth_map[i, j]
	    			
	    			break
	    	if m==True:
	    		break
    
    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    #print("Center_rel: ", depth_map[320, 240])
    #print("Obj_rel: ", depth_map[330, 500])


    #Get rid of points with value 0 (i.e no depth)
    mask_map = depth_map > 0

    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    end = time.time()
    totalTime = end - start
    #print("Total time: ", totalTime)

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
    
    start_point = (0, 400)
    end_point = (640, 400)

    # Draw the line on the image
    cv2.line(img, start_point, end_point, (255, 0, 0), 2)
    #cv2.circle(img, (320, 240), 10, (0,0,255), -1)
    #cv2.circle(img, (500, 330), 10, (0,0,255), -1)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    
    if k:
    	stat="0"
    	rospy.init_node('stop', anonymous=True)
    	rate = rospy.Rate(10)
    	pub.publish(stat)
    	rate.sleep()
    	print("Right")
    if m:
    	stat="2"
    	rospy.init_node('stop', anonymous=True)
    	rate = rospy.Rate(10)
    	pub.publish(stat)
    	rate.sleep()
    	print("Left")
    if r:
    	stat="1"
    	rospy.init_node('stop', anonymous=True)
    	rate = rospy.Rate(10)
    	pub.publish(stat)
    	rate.sleep()
    	print("Center")
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

