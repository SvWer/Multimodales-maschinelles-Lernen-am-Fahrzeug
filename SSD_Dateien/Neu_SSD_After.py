#  watch nvidia-smi

'''
    Diese Datei führt den SSD Algorithmus auf einem Datensatz aus. Dabei kann ganz unten gewählt werden, 
    ob zusätzlich der SSD ausgeführt werden soll, wenn vorher auf das Bild eine Maske aus dem Tiefenbild 
    drauf gelegt wurde.

'''
import numpy as np
import numpy.random as npr
import os
import sys
import tensorflow as tf
import cv2
import json
import lgsvl
import random
import time
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mping

import collections
import itertools

MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

#PATH_TO_CKPT = '/home/sven/Dokumente/Sven/Dateien/models/research/object_detection/ssd_mobilnet_v1_coco_2018_01_28/frozen_inference_graph.pb'
#PATH_TO_CKPT = './ssd_mobilnet_v1_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#load image helper function
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    
def load_image_into_numpy_array1(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)
    
    
PATH = '/home/sven/Dokumente/Sven/Dateien/images/'
#PATH = '/home/sven/Dokumente/Sven/Dateien/tests/'
#PATH = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten003_243/'
#PATH = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten010_8641/'
#PATH = '/home/sven/Dokumente/Sven/Dateien/Daten008/'
PATH_TO_IMAGES = PATH + 'main_img/'
PATH_TO_DEPTH_IMAGES = PATH + 'depth_img/'

img_count = len([name for name in os.listdir(PATH_TO_IMAGES) if os.path.isfile(os.path.join(PATH_TO_IMAGES, name))]) +1
depth_img_count = len([name for name in os.listdir(PATH_TO_DEPTH_IMAGES) if os.path.isfile(os.path.join(PATH_TO_DEPTH_IMAGES, name))]) +1

#IMAGE_PATHS = [ os.path.join(PATH_TO_IMAGES, 'main_{}.jpg'.format(i))for i in range(1, img_count)]
IMAGE_PATHS = [ os.path.join(PATH_TO_IMAGES, 'main_{}.png'.format(i))for i in range(1, img_count)]
#DEPTH_IMAGE_PATHS = [ os.path.join(PATH_TO_DEPTH_IMAGES, 'depth_{}.jpg'.format(i))for i in range(1, depth_img_count)]
DEPTH_IMAGE_PATHS = [ os.path.join(PATH_TO_DEPTH_IMAGES, 'depth_{}.png'.format(i))for i in range(1, depth_img_count)]

json_obj = {
	"images": []
}

json_obj_after = {
	"images": []
}

json_obj_own = {
	"images": []
}

json_obj_before = {
	"images": []
}

height = 480
width = 720
near = 0.1
far = 300.0

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options) 
#config.gpu_options.allow_growth = True

'''
    Funktion that calculates the Angle between the middle and an other height-value. 
    Uses a image height of 480pixels
    Param Y: Gets an Pixelvalue of Height
    
    Return:    Angle between Pixelvalue and half of the height
'''
def getAngle(y):
    return 30/240 *(y- 480/2)

def calcHeights(data, number):
    heights = np.zeros(shape=(480, 720))
    #for every Pixel: First Calculate Angle, then height from camera.
    for y in range(2, len(data)-2):
        for x in range(2, len(data[0])-2):
            alpha = getAngle(y)
            d = 1/(pow(data[y][x]/255, 10) * (1-float(far)/float(near))) - float(far)/(float(near)-float(far))
            h = np.sin(math.radians(alpha)) * d
            #Create Mask if height is between 2.5 and 6 and it is not as far away as 40 and all 15 pixel arround, so it maybe works better
            if h>2.5 and h < 6 and d < 40:
                heights[y-2][x-2] = 1
                heights[y-2][x-1] = 1
                heights[y-2][x] = 1
                heights[y-2][x+1] = 1
                heights[y-2][x+2] = 1
                heights[y-1][x-2] = 1
                heights[y-1][x-1] = 1
                heights[y-1][x] = 1
                heights[y-1][x+1] = 1
                heights[y-1][x+2] = 1
                heights[y][x-2] = 1
                heights[y][x-1] = 1
                heights[y][x] = 1
                heights[y][x+1] = 1
                heights[y][x+2] = 1
                heights[y+1][x-2] = 1
                heights[y+1][x-1] = 1
                heights[y+1][x] = 1
                heights[y+1][x+1] = 1
                heights[y+1][x+2] = 1
                heights[y+2][x-2] = 1
                heights[y+2][x-1] = 1
                heights[y+2][x] = 1
                heights[y+2][x+1] = 1
                heights[y+2][x+2] = 1
    #Save Mask-Image
    #matplotlib.image.imsave(SSD_PATH+'Masks/' + 'heights2_'+number+".jpg", heights)
    return heights

	
def performSSDAfter(PATHS, output_path, json_dic, json_path, delta):	 
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
      '''
        Creating Graph for Tensorboard, can be shown with: 
        tensorboard --logdir /tmp/test/1
      '''
      #writer = tf.compat.v1.summary.FileWriter("/tmp/test/1")
      #writer.add_graph(sess.graph)
      for image_path in PATHS:
        image = Image.open(image_path)
        temp_name = image_path.split('/')[8]
        temp_name = temp_name.split('.')[0]
        imgNumber = temp_name.split('_')[1]
        
        image_np = load_image_into_numpy_array(image)
            
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        
        scores_between = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
        scores_name = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
        boxes_between = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
        boxes_name = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
        
        # Actual detection.
        (boxes_between, scores_between) = sess.run(
            [boxes_name, scores_name],
            feed_dict={image_tensor: image_np_expanded})
            
        temp_name = image_path.split('/')[8]
        temp_name = temp_name.split('.')[0]
        imgNumber = temp_name.split('_')[1]
        
        im = Image.open(PATH_TO_DEPTH_IMAGES+'depth_'+imgNumber+'.png')
        
        '''
            Now after TF use Mask to split boxes. Use only boxes in Mask
        '''
        #delta = 0.2
        #depth_img = Image.open(PATH_TO_DEPTH_IMAGES+'depth_'+imgNumber+'.jpg')
        depth_img = Image.open(PATH_TO_DEPTH_IMAGES+'depth_'+imgNumber+'.png')
        mask = calcHeights(np.asarray(depth_img), 1)
        for i in range(len(scores_between[0])):
          if mask[int(boxes_between[0][i][0][0]*480)][int(boxes_between[0][i][0][1]*720)] == 1:
            scores_between[0][i][10] += delta
        
        
        '''
        Do the rest again with Tensorflow
        '''
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded, boxes_name: boxes_between, scores_name: scores_between})
         
            
        # Visualization of the results of a detection.
        #vis_util.visualize_boxes_and_labels_on_image_array(
        #    image_np,
        #    np.squeeze(boxes),
        #    np.squeeze(classes).astype(np.int32),
        #    np.squeeze(scores),
        #    category_index,
        #    use_normalized_coordinates=True,
        #    min_score_thresh=0.01,
        #    line_thickness=8)
        

        i = 0
        print("")	  
        print(image_path)
        print(image_path.split('/')[7])	
        print("Image size (Height, Width, Channels):", image_np.shape)
        print("-------------------------------")	  

        curr_obj_1 = {
                "ImageName" : image_path.split('/')[8],
                "Image" : int(imgNumber),
                "ImageShape": image_np.shape,
                "NumberDetections": int(np.squeeze(num_detections)),
                "Classes" : []
		}
       
        while (i < len(np.squeeze(scores))):
           currentScore = np.squeeze(scores)[i]
           if currentScore >= 0.01:
               currentClasses = np.squeeze(classes).astype(np.int32)[i]
               print("Object:", currentClasses)
               print("Score:", currentScore)
               print("Box (ymin, xmin, ymax, xmax):", np.squeeze(boxes)[i])
               print("Classes:", np.squeeze(classes)[i])
               print("Number Detection:", np.squeeze(num_detections))
               print("")

               _boxes = np.squeeze(boxes)[i]
               curr_obj_2 = {			    
				    "ClassName": str(category_index[currentClasses]["name"]),
                    "Class": int(np.squeeze(classes)[i]),
                    "Box": [float(_boxes[0]), float(_boxes[1]), float(_boxes[2]), float(_boxes[3])], 
                    "Score": float(currentScore)			   
                }
               curr_obj_1["Classes"].append(curr_obj_2)
               json_dic["images"].append(curr_obj_1)

               #im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
               #name = output_path + "img_"+str(imgNumber)+ ".jpg"
               #cv2.imwrite(name, im_rgb);	           
           i += 1

        result_json = []    
        
        for j in json_dic["images"]:
            if j not in result_json:
                result_json.append(j)

        json_dic["images"] = result_json

        with open(json_path, 'w', encoding='utf-8') as f:
              json.dump(json_dic, f, ensure_ascii=False, indent=4)
			  
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


SSD_PATH = PATH + 'SSD_Detections/'

'''
    Perform SSD with Mask after SSD
'''
count = [ 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
for i in count:
  json_obj_after = {
    "images": []
  }
  start_time = time.time()  
  performSSDAfter(IMAGE_PATHS, SSD_PATH, json_obj_after, SSD_PATH+'dataAfterMask'+str(i)+'.json',i)
  print("Object Detection with applied mask after SSD was running for ",(time.time() - start_time),"seconds.")
  start_time = time.time()