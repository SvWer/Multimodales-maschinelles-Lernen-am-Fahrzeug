#!/usr/bin/env  python3

# roscore
# roslaunch rosbridge_server rosbridge_websocket.launch
# rosrun subscriber_node sub2D.py
# python3 saveImgs_sven.py
# 

import rospy
import math
import message_filters
import numpy as np

from lgsvl_msgs.msg import Detection2DArray, Detection3DArray, SignalArray
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

image_path = "/home/sven/Dokumente/Sven/Dateien/images/"
i = 1

def alles_callback(msg2d, msg3d, msgdepth, msgSeg, msgMain, lid):
  
  global i
  localI = i 
  i += 1
  
  
  depth = np.fromstring(msgdepth.data, np.uint8)
  image_np = cv2.imdecode(depth, 0)
  cv2.imwrite(image_path+"depth_img/"+"depth_"+ str(localI) +".jpg", image_np)
  
  #Save segmentation image
  segm = np.fromstring(msgSeg.data, np.uint8)
  image_np = cv2.imdecode(segm, 1)
  cv2.imwrite(image_path+"seg_img/"+"segm_"+ str(localI) +".jpg", image_np)
  
   #Save Main image
  main = np.fromstring(msgMain.data, np.uint8)
  image_np = cv2.imdecode(main, -1)
  cv2.imwrite(image_path+"main_img/"+"main_"+ str(localI) +".jpg", image_np)
  
  #Save 2D ground Truth
  f2D = open(image_path+"2DGroundTruth/"+"2dgT_"+str(localI)+".txt", "a")
  f2D.write(str(msg2d.detections))
  f2D.close()
  
  #Save 2D ground Truth
  f3D = open(image_path+"3DGroundTruth/"+"3dgT_"+str(localI)+".txt", "a")
  f3D.write(str(msg3d.detections))
  f3D.close()
  
  #Save Lidar
  lD = open(image_path+"lidar/"+"lidar_"+str(localI)+".pcd", "a")
  lD.write(str(lid.data))
  lD.close()
  
  
if __name__ == '__main__':
    rospy.init_node('listentomsgs', anonymous=True)
    gT2D = message_filters.Subscriber('/simulator/ground_truth/2d_detections', Detection2DArray)
    gT3D = message_filters.Subscriber('/simulator/ground_truth/3d_detections', Detection3DArray)
    depth = message_filters.Subscriber('/simulator/depth_camera/compressed', CompressedImage)
    segm = message_filters.Subscriber('/simulator/segmentation_camera/compressed', CompressedImage)
    mainCam = message_filters.Subscriber('/simulator/main_camera/compressed', CompressedImage)
    lidar = message_filters.Subscriber('/simulation/lidar/PointCloud2',PointCloud2)
    #traffL = message_filters.Subscriber('/simulator/ground_truth/trafficLights', SignalArray)

    ts = message_filters.ApproximateTimeSynchronizer([gT2D, gT3D, depth, segm, mainCam, lidar], 100, 0.1)
    ts.registerCallback(alles_callback)


    rospy.spin()




# rostopic echo simulator/ground_truth/2d_dections
